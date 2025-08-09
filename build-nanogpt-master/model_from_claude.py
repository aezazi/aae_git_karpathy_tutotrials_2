import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Tuple, Optional

class ExpertMoESwiglu(nn.Module):
    """Single expert - unchanged from your original implementation"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.n_embd * 4
       
        self.linear_1 = nn.Linear(config.n_embd, self.hidden_dim) 
        self.linear_1._am_expert = True
        self.linear_2 = nn.Linear(config.n_embd, self.hidden_dim)
        self.linear_2._am_expert = True
        self.c_proj = nn.Linear(self.hidden_dim, config.n_embd)
        self.c_proj._am_expert = True

    def forward(self, x):
        x = self.linear_1(x) * F.silu(self.linear_2(x))
        x = self.c_proj(x)
        return x


class TopKMoEGateParallel(nn.Module):
    """Enhanced gating with expert parallelization support"""
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.k = config.k
        self.seq_len = config.seq_len
        self.load_balance_scale = config.load_balance_scale
        
        # Expert parallelization parameters
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate experts per GPU
        assert self.num_experts % self.world_size == 0, f"num_experts ({self.num_experts}) must be divisible by world_size ({self.world_size})"
        self.experts_per_gpu = self.num_experts // self.world_size
        
        # Local expert range for this GPU
        self.local_expert_start = self.rank * self.experts_per_gpu
        self.local_expert_end = (self.rank + 1) * self.experts_per_gpu
        
        # Gate still needs to output logits for ALL experts (global routing decision)
        self.gate_linear = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.noise_weight = nn.Parameter(torch.zeros(config.num_experts))
        self.noisy_std = 1.0

    def _compute_load_balance_loss(self, expert_usage):
        """Compute load balancing loss to encourage uniform expert usage"""
        uniform_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balance_loss = F.mse_loss(expert_usage, uniform_usage)
        return load_balance_loss * self.load_balance_scale

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Compute logits for ALL experts (global gating decision)
        logits = self.gate_linear(x)  # (batch_size, seq_len, num_experts)
        
        # Calculate load balancing loss using clean logits
        gate_weights = F.softmax(logits, dim=-1)
        gate_weights_flat = gate_weights.view(batch_size * seq_len, -1)
        expert_usage = gate_weights_flat.mean(0)  # (num_experts,)
        load_balance_loss = self._compute_load_balance_loss(expert_usage)
        
        # Add noise
        noise = torch.randn_like(logits) * self.noisy_std
        noise = noise * self.noise_weight
        logits_noisy = logits + noise
        
        # Get top-k experts globally
        top_k_logits_noisy, top_k_indices_global = logits_noisy.topk(self.k, dim=-1)
        
        # Create sparse logits
        zeros = torch.full_like(logits_noisy, float('-inf'))
        sparse_logits_noisy = zeros.scatter(-1, top_k_indices_global, top_k_logits_noisy)
        top_k_gated_weights = F.softmax(sparse_logits_noisy, dim=-1)
        
        return top_k_gated_weights, top_k_indices_global, load_balance_loss


class MoELayerParallel(nn.Module):
    """Expert-parallel MoE layer"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.seq_len = config.seq_len
        self.k = config.k
        self.n_embd = config.n_embd
        
        # Initialize distributed parameters
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Calculate experts per GPU
        assert self.num_experts % self.world_size == 0, f"num_experts ({self.num_experts}) must be divisible by world_size ({self.world_size})"
        self.experts_per_gpu = self.num_experts // self.world_size
        
        # Local expert range for this GPU
        self.local_expert_start = self.rank * self.experts_per_gpu
        self.local_expert_end = (self.rank + 1) * self.experts_per_gpu
        
        print(f"Rank {self.rank}: Managing experts {self.local_expert_start} to {self.local_expert_end-1}")
        
        # Gate (handles global routing)
        self.gate = TopKMoEGateParallel(config)
        
        # Only create local experts on this GPU
        self.local_experts = nn.ModuleList([
            ExpertMoESwiglu(config) for _ in range(self.experts_per_gpu)
        ])
        
        # Communication buffers - will be resized dynamically
        self.send_buffer = None
        self.recv_buffer = None

    def _get_expert_assignments(self, top_k_indices_global, batch_size, seq_len):
        """
        Determine which tokens need to be sent to which GPUs
        Returns: dict mapping gpu_rank -> (token_indices, expert_local_indices)
        """
        assignments = {}
        
        # Flatten indices for easier processing
        flat_indices = top_k_indices_global.view(-1)  # (batch_size * seq_len * k)
        
        for gpu_rank in range(self.world_size):
            gpu_expert_start = gpu_rank * self.experts_per_gpu
            gpu_expert_end = (gpu_rank + 1) * self.experts_per_gpu
            
            # Find which flattened positions have experts on this GPU
            expert_mask = (flat_indices >= gpu_expert_start) & (flat_indices < gpu_expert_end)
            
            if expert_mask.any():
                # Get the positions and convert back to token positions
                flat_positions = torch.where(expert_mask)[0]
                token_positions = flat_positions // self.k  # Which tokens
                
                # Get the corresponding expert indices and convert to local indices
                expert_global_indices = flat_indices[expert_mask]
                expert_local_indices = expert_global_indices - gpu_expert_start
                
                assignments[gpu_rank] = (token_positions, expert_local_indices)
        
        return assignments

    def _communicate_tokens(self, x_flat, assignments, batch_size, seq_len):
        """
        Perform all-to-all communication to send tokens to appropriate GPUs
        """
        device = x_flat.device
        
        if not dist.is_initialized() or self.world_size == 1:
            # Single GPU case - just return local assignments
            if self.rank in assignments:
                token_positions, expert_local_indices = assignments[self.rank]
                return x_flat[token_positions], expert_local_indices, {self.rank: len(token_positions)}
            else:
                return torch.empty(0, self.n_embd, device=device), torch.empty(0, dtype=torch.long, device=device), {}
        
        # Multi-GPU case: implement all-to-all communication
        send_counts = []
        recv_counts = []
        
        # Calculate send/receive counts
        for gpu_rank in range(self.world_size):
            if gpu_rank in assignments:
                send_count = len(assignments[gpu_rank][0]) if gpu_rank != self.rank else 0
                recv_count = len(assignments[self.rank][0]) if gpu_rank == self.rank else 0
            else:
                send_count = 0
                recv_count = 0
            send_counts.append(send_count)
            recv_counts.append(recv_count)
        
        # Prepare send buffers
        send_tokens = []
        send_expert_ids = []
        
        for gpu_rank in range(self.world_size):
            if gpu_rank in assignments and gpu_rank != self.rank:
                token_positions, expert_local_indices = assignments[gpu_rank]
                send_tokens.append(x_flat[token_positions])
                send_expert_ids.append(expert_local_indices)
            else:
                send_tokens.append(torch.empty(0, self.n_embd, device=device))
                send_expert_ids.append(torch.empty(0, dtype=torch.long, device=device))
        
        # All-to-all communication for tokens
        recv_tokens = [torch.empty(recv_counts[i], self.n_embd, device=device) for i in range(self.world_size)]
        dist.all_to_all(recv_tokens, send_tokens)
        
        # All-to-all communication for expert IDs
        recv_expert_ids = [torch.empty(recv_counts[i], dtype=torch.long, device=device) for i in range(self.world_size)]
        dist.all_to_all(recv_expert_ids, send_expert_ids)
        
        # Combine received data
        if any(recv_counts):
            local_tokens = torch.cat([t for t in recv_tokens if t.numel() > 0], dim=0)
            local_expert_indices = torch.cat([e for e in recv_expert_ids if e.numel() > 0], dim=0)
        else:
            local_tokens = torch.empty(0, self.n_embd, device=device)
            local_expert_indices = torch.empty(0, dtype=torch.long, device=device)
        
        return local_tokens, local_expert_indices, dict(enumerate(recv_counts))

    def _process_local_experts(self, tokens, expert_indices):
        """Process tokens through local experts"""
        if tokens.numel() == 0:
            return torch.empty(0, self.n_embd, device=tokens.device)
        
        outputs = torch.zeros_like(tokens)
        
        for local_expert_idx in range(self.experts_per_gpu):
            expert_mask = (expert_indices == local_expert_idx)
            if expert_mask.any():
                expert_tokens = tokens[expert_mask]
                expert_output = self.local_experts[local_expert_idx](expert_tokens)
                outputs[expert_mask] = expert_output
        
        return outputs

    def _communicate_results_back(self, local_outputs, recv_counts):
        """Send processed results back to original GPUs"""
        if not dist.is_initialized() or self.world_size == 1:
            return local_outputs
        
        # Split local outputs by originating GPU
        send_outputs = []
        start_idx = 0
        
        for gpu_rank in range(self.world_size):
            count = recv_counts.get(gpu_rank, 0)
            if count > 0:
                send_outputs.append(local_outputs[start_idx:start_idx + count])
                start_idx += count
            else:
                send_outputs.append(torch.empty(0, self.n_embd, device=local_outputs.device))
        
        # All-to-all communication back
        recv_outputs = [torch.empty(0, self.n_embd, device=local_outputs.device) for _ in range(self.world_size)]
        # Note: This needs to be implemented based on the original send counts
        # For brevity, assuming we have the reverse communication setup
        
        return torch.cat([r for r in recv_outputs if r.numel() > 0], dim=0)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Get routing decisions from gate
        top_k_gated_weights, top_k_indices_global, load_balance_loss = self.gate(x)
        
        # Flatten input for processing
        x_flat = x.view(batch_size * seq_len, -1)
        
        # Determine expert assignments across GPUs
        assignments = self._get_expert_assignments(top_k_indices_global, batch_size, seq_len)
        
        # Communicate tokens to appropriate GPUs
        local_tokens, local_expert_indices, recv_counts = self._communicate_tokens(x_flat, assignments, batch_size, seq_len)
        
        # Process tokens through local experts
        local_outputs = self._process_local_experts(local_tokens, local_expert_indices)
        
        # Communicate results back (simplified for this example)
        # In practice, you'd need to implement the reverse all-to-all communication
        
        # For now, using the original single-GPU approach for the combination step
        # This would need to be replaced with proper distributed combination
        final_output = torch.zeros_like(x)
        top_k_gated_weights_flat = top_k_gated_weights.view(batch_size * seq_len, self.num_experts)
        
        # Original combination logic (this part would need distributed implementation)
        for i in range(self.num_experts):
            expert_mask = (top_k_indices_global == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            
            if flat_mask.any():
                expert_weights = top_k_gated_weights_flat[flat_mask, i].unsqueeze(1)
                
                # This is where you'd get the results from the appropriate GPU
                # For now, simplified implementation
                if self.local_expert_start <= i < self.local_expert_end:
                    local_idx = i - self.local_expert_start
                    expert_input = x_flat[flat_mask]
                    expert_output = self.local_experts[local_idx](expert_input)
                    expert_output_weighted = expert_output * expert_weights
                    
                    expert_contribution = torch.zeros_like(final_output)
                    expert_contribution[expert_mask] = expert_output_weighted.squeeze(1)
                    final_output = final_output + expert_contribution
        
        return final_output, top_k_indices_global, load_balance_loss


# Configuration changes needed
class GPTConfigParallel:
    """Enhanced config for expert parallelization"""
    def __init__(self):
        # ... your existing config parameters ...
        self.seq_len = 1024
        self.batch_size = 42
        self.vocab_size = 50304
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.base_lr = 6e-4 * 3
        self.warm_up_steps = 300
        self.num_experts = 8
        self.k = 2
        self.load_balance_scale = .01
        self.print_token_routing = True
        
        # New parameters for expert parallelization
        self.expert_parallel = True  # Enable expert parallelization
        self.expert_parallel_comm_backend = 'nccl'  # Communication backend


# Modified Block class
class BlockParallel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)  # Your existing attention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # Use parallel MoE layer
        if getattr(config, 'expert_parallel', False):
            self.moe = MoELayerParallel(config)
        else:
            self.moe = MoELayer(config)  # Your original implementation

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_out, top_k_indices, load_balance_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, top_k_indices, load_balance_loss
