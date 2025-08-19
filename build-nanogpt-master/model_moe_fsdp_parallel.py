#%%
"""
Model definition
"""
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from hellaswag import render_example, iterate_examples

#%%
class RotaryPosEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        # self.get_theta()
        theta =  10_000 ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        self.register_buffer("theta", theta)
        position = torch.arange(0, config.seq_len, 1.0)
        angles = torch.outer(position, self.theta)
        self.register_buffer("cached_angles", angles)  # (max_seq_len, head_dim // 2)
        

    # Note that the code below allows for variable length sequences. If sequence length is always fixed, it would be more efficient to compute angles in the __init()__ and resigter to a buffer as I have done above. I'm leaving this function here for reference
    # def get_angles(self, seq_len=1024, device=None):
    #     position = torch.arange(0, seq_len, 1.0)
    #     angles = torch.outer(position.to(device=device), self.theta)
    #     return angles
    
    # x is the input vector with shape: [batch_size, seq_length, num_heads, head_dim]
    def apply_rotation(self, x=None):
        
        device = x.device
        seq_len = x.shape[1]

        angles = self.cached_angles[:seq_len].to(device)
        #
        #  Apply sin and cos to angles and use unsqueeze to add dimensions to match number of dimensions of input vector 
        sin = angles.sin().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
        cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]

        # split input vector x into two vectors from the  even and odd indexed elements of the original vector x. Each element from x1 and x2 will be paired for rotation
        x1 = x[:, :, :, 0: :2]
        x2 = x[:, :, :, 1: :2]

        # Apply rotation. Note that the elementwise multiplication broadcasts the sin and cos values into batch and num_heads dimensions
        x1_rot = x1 * cos - x2 * sin #[B, S, num_heads,  head_dim/2]
        x2_rot = x1 * sin + x2 * cos #[B, S, num_heads,  head_dim/2]

        # Stack into [B, S, head_num, head_dim/2, 2] the dim=-1 adds a new dimension to the end of [B, S, H, head_dim/2] and stacks each corresponding element from dim=1 of [seq_length, dim/2] from the x_rotated_even and x_rotated_odd vectors into that new third dimension
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1) #[B, S, H, head_dim/2, 2]

        # flatten last two dims back to [B, seq_len, num_head, head_dim]
        x_rot = x_rot.flatten(start_dim=3, end_dim=-1)
        
        return x_rot

#%%
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 

        # key, query, value, projections for all heads, but in a batch. The output of the linear layer is 3 times the size of the input. The 3x multiplication is because we later divide the output of the linear layer into 3 vectors for q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # attribute flag to id c_proj during weight initialization
        self.rot_embed = RotaryPosEmbed(config) # rotary embedding
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # input is a batch of sequences of embeddings
        B, seq_len, n_embd = x.size()

        # split the embeddings into key, query, value
        # B is batch size. seq_len is the length of each sequence. the last dimension is the embedding dimension. n_heads is number of heads, dim_heads is the number of dimensions of each head, and self.n_embd is the dimensionality of the original input embedding and model residual stream. n_embd = n_heads * dim_heads  e.g. in GPT-2 (124M), n_heads=12, dim_heads=64, so n_heads*dim_heads=n_embd=768 channels in the transformer. Note from above the self.c_attn() projects the dimensionality(n_embd) of the input by 3x so that when split into q, k, v vectors, each will have dim = n_embd. 
        qkv = self.c_attn(x) # (B, seq_len, 3 * n_embd)
        # print(qkv.shape)

        # split the output of the linear layer into 3 vectors for q, k, v
        q, k, v = qkv.chunk(3, dim=-1) # each has shape (B, seq_len, n_embd)


        # Karpathy explains the purpose of the following to be to make the training process more efficient in Pytorch by splitting the channels into multiple heads. Each head is a slice of the channels. This allows for more parallelization and less memory usage.
    
        # for rotary embedding, do not tranpose k and q to (B, n_heads, seq_len, dim_heads) until the rotation is applied
        k = k.view(B, seq_len, self.n_head, n_embd // self.n_head) # (B, seq_len, n_heads, dim_heads)
        q = q.view(B, seq_len, self.n_head, n_embd // self.n_head) # (B, seq_len, n_heads, dim_heads)
    
        # apply rotation and transpose. the reshaping to (B, n_heads, seq_len, dim_heads) is to accommodate the matrix multiplication of k, q, and v along the desired dimensions
        k_rot = self.rot_embed.apply_rotation(x=k).transpose(1, 2) # (B, n_heads, seq_len, dim_heads)
        q_rot = self.rot_embed.apply_rotation(x=q).transpose(1, 2) # (B, n_heads, seq_len, dim_heads)
        
        v = v.view(B, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (B, n_heads, seq_len, dim_heads)


        # Pytorch implementation of Flash attention algorithim. This is the scaled dot-product attention built-in pytorch function. It takes the dot product of the query and key, scales it by the square root of the head size, and then applies a softmax to get the attention weights. The attention weights are then multiplied by the value to get the output. the is_causal=True argument ensures that the attention is only applied to the left of the current position in the sequence (i.e. it is causal). This is done by applying a mask to the attention weights. See Karpathy's video tutorial at 2:00:00 for more details. 
        y = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True) # (B, n_heads, seq_len, dim_heads)
        
        # transpose back to (B, seq_len, n_heads*dim_heads) and combine heads. Note that the y vector returned by scaled_dot_product is not contiguous. For view() to work, the original tensor must be contiguous in memory. reshape() can work with both contiguous and non-contiguous tensors, automatically handling the necessary memory operations. 
        y = y.transpose(1, 2).contiguous().view(B, seq_len, n_embd)

        y = self.c_proj(y)
        return y


#%%
# MoE feedforward network
#This class creates a single expert in the MoE layer. Each expert is a simple feedforward network with a swiglu activation function.
class ExpertMoESwiglu(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.n_embd * 4 # hidden dimension for the expert
       
        self.linear_1 = nn.Linear(config.n_embd, self.hidden_dim) 
        self.linear_1._am_expert =True
        self.linear_2 = nn.Linear(config.n_embd, self.hidden_dim)
        self.linear_2._am_expert =True
        self.c_proj = nn.Linear(self.hidden_dim, config.n_embd)
        self.c_proj._am_expert =True
        # torch.manual_seed(42)

    def forward(self, x):
        x =self.linear_1(x) * F.silu(self.linear_2(x))  # this is Swiglu activation
        x= self.c_proj(x)
        return x
    
# this class implemets top_k sparse gating 
class TopKGateParallel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.k = config.k
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.load_balance_scale = config.load_balance_scale

        # expert parallelization parameters
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # compute num experts per gpu
        assert self.num_experts % self.world_size == 0, f'num_experts ({self.num_experts}) must be divisible by world size ({self.world_size})'

        # local expert range for this gpu. Clever approach suggested by Claude for indexing local gpus to match global ids. Note that this generates ids for input into range(). So if there are 4 experts per gpu, for rank=0 ,this logic produces start=0, end=4. When input into range(0,4) will produce ids 0,1,2,3. For rank=2, start=8 end=12 range(8,12) will produce ids 8,9,10,11
        self.experts_per_gpu = self.num_experts // self.world_size
        self.local_expert_id_start = self.rank * self.experts_per_gpu
        self.local_expert_id_end = (self.rank +1) * self.experts_per_gpu 

        # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each expert. the logits will have shape (batch_size, seq_len, num_experts) 
        self.gate_linear = nn.Linear(self.n_embd, self.num_experts, bias=False)

        # this is a learnable parameter that will be used to scale the noise added to the logits of each expert. The allows the noise added to each expert to be "customized" and dynamic. It adapts during training depending on the expert. It is initialized to zero and will be learned during training.
        self.noise_weight = nn.Parameter(torch.zeros(config.num_experts)) 

        # this is the standard deviation of the noise added to the logits of each expert. It is a hyperparameter that can be tuned. Note that unlike the noise_weight it is a global parameter, this is not a learnable parameter.
        self.noisy_std = 1.0

    def _compute_load_balance_loss(self, gate_weights_mean):
        # the code here was suggested by Claude
        """Compute load balancing loss to encourage uniform expert usage"""
        
        # Method 1: Variance-based (simpler)
        uniform_usage = torch.ones_like(gate_weights_mean) / self.num_experts
        load_balance_loss = F.mse_loss(gate_weights_mean, uniform_usage)
        
        # Method 2: Entropy-based (more principled)
        # load_balance_loss = -torch.sum(gate_weights_mean * torch.log(gate_weights_mean + 1e-8))
        
        # Method 3: Coefficient of variation (Switch Transformer style)
        # mean_usage = gate_weights_mean.mean()
        # std_usage = gate_weights_mean.std()
        # load_balance_loss = std_usage / (mean_usage + 1e-8)
        
        return load_balance_loss * self.load_balance_scale
    
    def forward(self, x_flat):
        # print(f'x shape: {x.shape}')
        # x has shape (batch_size*sequence_length, embedding dimension) and is the output of the multi-head attention layer. 
        token_count, n_embd = x_flat.shape
        
        assert n_embd == self.n_embd, f"Expected embedding dim {self.n_embd}, got {n_embd}"
        assert token_count == self.batch_size*self.seq_len, f"Expected embedding dim {self.batch_size*self.seq_len}, got {token_count}"
        
        # project x_flat to ()
        logits = self.gate_linear(x_flat) # (batch_size*seq_len, num_experts)

        # compute load balancing loss using clean logits before noise and topk
        gate_weights = F.softmax(logits, dim=-1)
        # print(f'gate_weights shape: {gate_weights.shape}')
        
        gate_weights_mean= gate_weights.mean(0)  # (num_experts,) the mean of gate_weights over all tokens
        load_balance_loss = self._compute_load_balance_loss(gate_weights_mean)
        # print(f'load balancing loss: {load_balance_loss}')

        # The global noise to be added to the logits of each expert. This is a random noise tensor of the same shape as the logits multiplied by noisy_std to create desired level of variance
        noise = torch.randn_like(logits) * self.noisy_std # (batch_size, seq_len, num_experts)

        # Per-expert noise scaling using self.noise_weights. 
        noise = noise * self.noise_weight

        # Add the noise to the logits. 
        logits_noisy = logits + noise  # (batch_size, seq_len, num_experts)

         # Get the top-k logits and their corresponding indices. The pytorch top_k method returns the top-k values and their indices along the last dimension (num_experts). In each batch, for each token in the sequence, it selects the top-k logits and their indices from the logits produced by each expert. Note that the top_k ids are global and not necessarily on this gpu.
        top_k_logits_noisy, top_k_ids_global_noisy = logits_noisy.topk(self.k, dim=-1)  # (batch_size, seq_len, top_k) 

        # We want sparse matrices. We achieve this by keeping the top-k logits for each token and filling the rest with a value that represents "no contribution" (like negative infinity).  This is done to ensure that the softmax function will ignore these values when computing the weights for each expert. So only the values produced by the top-k experts will contribute to the final output. We implement this by first creating a tensor of the same shape as the logits filled with negative infinity, and then using the scatter function to fill in the top-k logits at the indices of the top-k experts. Note that in this context, softmax is being used to compute "weights" for each expert not probabilities as for multiclass classification. Its a subttle difference but I think important to note.
        
        #full_like clones a tensor and fills it with a specified value (like infinity).
        zeros = torch.full_like(logits_noisy, float('-inf')) 

        # scatter fills the zeros tensor with the top_k_logits at the indices of the top_k_ids_global_noisy along the last dimension (-1). This creates a sparse matrix where only the top-k logits are kept and the rest are filled with negative infinity.
        sparse_logits_noisy = zeros.scatter(-1, top_k_ids_global_noisy, top_k_logits_noisy)

        # Note top_k_gated_weights has shape #(B, seq_len, num_experts). The top_k experts will have weights that sum to 1, the other experts will have weights-inf
        top_k_gated_weights = F.softmax(sparse_logits_noisy, dim=-1) 

        return top_k_gated_weights, top_k_ids_global_noisy, load_balance_loss
    
class MoELayerParallel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.k = config.k
        self.n_embd = config.n_embd

        # expert parallelization parameters
        self.world_size = config.world_size
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.experts_per_gpu = config.experts_per_gpu

        # expert to gpu assginment based on gpu rank. Pre compute expert assginments for all gpus
        self.gpu_expert_ranges = {} # dictionary to hold gpu --> expert assignment
        for gpu_rank in range(self.world_size):
            start = gpu_rank * self.experts_per_gpu
            end = (gpu_rank+1) * self.experts_per_gpu
            self.gpu_expert_ranges[gpu_rank] = (start, end)

        # global expert id range for this gou
        self.local_expert_start, self.local_expert_end = self.gpu_expert_ranges[self.rank]
    

        print(f"Rank {self.rank}: Managing experts {self.local_expert_start} to {self.local_expert_end-1}")

        # create the top-k gate for this gpu
        self.gate = TopKGateParallel(config)

        # create local experts for this GPU
        self.local_experts = nn.ModuleList([
            ExpertMoESwiglu(config) for _ in range(self.experts_per_gpu)
        ])

        # create a tensor to hold a running count of the number of tokens processed by each expert on this gpu.
        self.count_tokens_processed_by_each_expert = torch.zeros(self.experts_per_gpu, dtype=torch.int, device=torch.device(f"cuda:{self.rank}"), requires_grad=False) # (num_experts_per_gpu,)


        # NOTE: all-to-all communication: In a MoE setup, the topk gate will assign tokens being processed on this gpu to anyone of the num_experts. The experts assgined to a token may or may not be on this gpu. So we have to identify which gpu is hosting the expert to which a token is assgined, send the token to that gpu/expert for processing  and then receive the result back to rejoin the batch-sequence on this gpu.  
        
        # create communication buffers for all to all communication.
        self.send_buffer = None
        self.receive_buffer = None
    
    def _get_expert_assignments_with_padding(self, top_k_ids_global, gate_weights_global):
        """
        determine which tokens are assgined to which gpu by the topk gate.
        returns a dictionary that maps gpu global rank --> (token_indices, expert_local_id)
        """
        assignments = {}
    
        # flatten topk expert ids to shape (B*T, k) tensor for easier processing. 
        # top_k_ids_global = top_k_ids_global.view((self.batch_size*self.seq_len), -1)

        # Note: gate_weights_global is already of shape (batch_size*seq_len, num_experts) so no need to flatten.

        for gpu_rank in range(self.world_size):
            # get the start and end expert ids assigned to each gpu
            gpu_expert_start, gpu_expert_end = self.gpu_expert_ranges[gpu_rank]

            # a mask to filter all tokens in the sequence with an assigned topk expert id on this gpu_rank. The mask is a 1D tensor that will return TRUE for any row (token) that contains an expert in the range of experts assigned to this gpu_rank and FALSE otherwise. Note that for tensors, we need to use bitwise comparison operators. So use "&" instead of "and"
            expert_mask = ((top_k_ids_global>=gpu_expert_start) & (top_k_ids_global < gpu_expert_end)).any(dim=-1) # (batch_size*seq_len,)

            if expert_mask.any():
                # We extract the positions in expert_mask with value TRUE. These positions correspond to the token positions (indicies) in the sequence that were assigned to experts on this gpu_rank. note that using torch.where(condition) with just one argument returns a tuple of 1-D tensors, where each tensor represents the indices of the elements in the input condition that evaluate to True along each dimension of the input tensor. The first element in the tuple corresponds to the row indices and the second element corresponds to the column indices. We need the row indices (element 0) which correspond to the token positions in the sequence.
                token_positions = torch.where(expert_mask)[0]

                # For each token which has one or both of its assigned experts on this gpu_rank, get its local expert assignments. The assginments may have different number of experts between 1 and k experts. all to all communication expects all tensors to be of the same size and dtype. So we have to pad all expert assignments to the same length (k). We also need to create a mask that we will eventually use to filter the padding.
                token_expert_local_id_assignments_padded = []
                token_weights_padded =[]
                
                for token_idx in token_positions:
                    # this returns the the topk (k=2 in this case) experts to which this token is assigned. One or both of the experts might be on this gpu_rank.
                    token_experts = top_k_ids_global[token_idx]


                    # create a mask to filter token experts on this gpu_rank. since one or both of the experts to which this token was assigned may be on this gpu_rank, we need to filter for only the experts on this gpu_rank. 
                    gpu_token_expert_mask = ((token_experts >= gpu_expert_start) & 
                                      (token_experts < gpu_expert_end))
                    
                    # apply the mask to get only token experts on this gpu_rank
                    token_experts_on_this_gpu = token_experts[gpu_token_expert_mask]

                    # use the token_experts_on_this_gpu to select from gate_weights_global, the weights for this token with an expert on this gpu_rank.
                    token_weights_on_this_gpu = gate_weights_global[token_idx][token_experts_on_this_gpu]

                    # the token expert ids we obtained above are global id range (0 - num_experts). we need to convert these to local ids since on each gpu the experts are indexed 0 to experts_per_gpu-1.
                    token_experts_local_ids = token_experts_on_this_gpu - gpu_expert_start

                    # pad the expert ids and weights to number of experts (k) so that all tensors are of the same size.
                    token_num_experts = len(token_experts_local_ids) # number of experts assigned to this token that are on this gpu_rank
                    if token_num_experts < self.k:
                        # if the number of experts is less than k, pad to length k with -1 to indicate dummy expert
                        token_expert_local_id_assignments_padded.append(F.pad(token_experts_local_ids, (0, self.k - token_num_experts), value=-1))

                        token_weights_padded.append(F.pad(token_weights_on_this_gpu, (0, self.k - token_num_experts), value=0.0))

                    else:
                        # if the number of experts is already k, no padding needed
                         token_expert_local_id_assignments_padded.append(token_experts_local_ids)  
                         
                         token_weights_padded.append(token_weights_on_this_gpu)

                    # by index, each element in token_expert_local_id_assignments_padded is the local id(s) of experts to which tokens in token_positions was assigned. So token at token_positions[3] was assigned to expert(s) at token_expert_local_id_assignments_padded[3]
                    

                # stack the padded experts ids into a tensor. This will have shape (num_tokens, k) where num_tokens is the number of tokens assigned to this gpu_rank and k is the number of experts per token.
                token_expert_local_id_assignments_padded = torch.stack(token_expert_local_id_assignments_padded)

                # stack the padded weights into a tensor same as done aove for token_expert_local_id_assignments_padded
                token_weights_padded = torch.stack(token_weights_padded)

                # create a mask to filter the padding. the mask will have shape (num_tokens, k) where num_tokens is the number of tokens assigned to this gpu_rank and k is the number of experts per token. The mask will be True for the experts that were assigned to the token and False for the padding dummy.
                token_expert_assignment_mask = (token_expert_local_id_assignments_padded >= 0)  # (num_tokens, k)
                
                # finally, package token_positions, their token_expert_local_id_assignments_padded, and token_expert_assignment_mask in a tuple and associate the tuple with this gpu_rank on which the experts reside. for example, assignments[1] carries the token positions and the experts to which the tokens were assigned and mask that can be processed by gpu rank=1
                assignments[gpu_rank] = (token_positions, token_expert_local_id_assignments_padded,
                token_weights_padded,
                token_expert_assignment_mask)
        
        return assignments
    
   
    def _communicate_tokens(self, x_flat, assignments):
        """
        Perform all-to-all communication to send tokens to appropriate GPUs
        """
        device = x_flat.device

        # prepare send data for each gpu
        send_tensors = []
        send_counts = []

        for gpu_rank in range(self.world_size):
            if gpu_rank in assignments:
                token_positions, token_expert_local_id_assignments_padded, token_weights_padded, token_expert_assignment_mask = assignments[gpu_rank]

                num_tokens = token_positions.shape[0]
                
                # extract the tokens with at least one expert on this gpu_rank
                tokens_to_send = x_flat[token_positions] # (num_tokens, n_embd)

                # package tokens, expert_ids, weights, and mask in a list for this gpu_rank
                send_tensors.extend([
                    tokens_to_send, 
                    token_expert_local_id_assignments_padded.to(device), 
                    token_weights_padded.to(device), 
                    token_expert_assignment_mask.to(device)
                ])

                send_counts.extend([
                    tokens_to_send.numel(),  # num elements in the token tensor (num_tokens * n_embd)
                    token_expert_local_id_assignments_padded.numel(),  # (num_tokens * k)
                    token_weights_padded.numel(),  # (num_tokens * k)
                    token_expert_assignment_mask.numel()  # (num_tokens * k)
                ])
            else:
                # if no tokens assigned to this gpu_rank, send empty tensors
                empty_tokens = torch.empty(0, self.n_embd, device=device, dtype=x_flat.dtype)
                empty_ids = torch.empty(0, self.k, device=device, dtype=torch.long)
                empty_weights = torch.empty(0, self.k, device=device, dtype=x_flat.dtype)
                empty_mask = torch.empty(0, self.k, device=device, dtype=torch.bool)
                
                send_tensors.extend([empty_tokens, empty_ids, empty_weights, empty_mask])
                send_counts.extend([0, 0, 0, 0])

        # prepare receive buffers
        recv_counts = [None] * self.world_size  # Initialize with None values
        
        # all_gather the counts - this is the fix!
        dist.all_gather_object(recv_counts, send_counts)
        
        # Now recv_counts is a list of lists, we need to flatten it
        recv_counts_flat = []
        for counts_from_rank in recv_counts:
            recv_counts_flat.extend(counts_from_rank)
        
        # Now use the flattened recv_counts for calculations
        # recv_counts_flat structure:
        # [GPU0_tokens, GPU0_experts, GPU0_weights, GPU0_mask,
        #  GPU1_tokens, GPU1_experts, GPU1_weights, GPU1_mask, ...]
                        
        # Compute total elements to receive for each tensor type
        total_token_elements = sum([recv_counts_flat[i*4] for i in range(self.world_size)])
        total_expert_elements = sum([recv_counts_flat[i*4 + 1] for i in range(self.world_size)])
        total_weight_elements = sum([recv_counts_flat[i*4 + 2] for i in range(self.world_size)])
        total_mask_elements = sum([recv_counts_flat[i*4 + 3] for i in range(self.world_size)])

        # Create 1D receive buffers for communication
        recv_tokens_flat = torch.empty(total_token_elements, device=device, dtype=x_flat.dtype)
        recv_expert_ids_flat = torch.empty(total_expert_elements, device=device, dtype=torch.long)
        recv_weights_flat = torch.empty(total_weight_elements, device=device, dtype=x_flat.dtype)
        recv_mask_flat = torch.empty(total_mask_elements, device=device, dtype=torch.bool)

        # flatten send tensors for all-to-all_single communication
        send_tokens_flat = torch.cat([t.flatten() for i, t in enumerate(send_tensors) if i % 4 == 0])
        send_expert_ids_flat = torch.cat([t.flatten() for i, t in enumerate(send_tensors) if i % 4 == 1])
        send_weights_flat = torch.cat([t.flatten() for i, t in enumerate(send_tensors) if i % 4 == 2])
        send_mask_flat = torch.cat([t.flatten() for i, t in enumerate(send_tensors) if i % 4 == 3])

        # Perform all-to-all communication to send tokens to appropriate GPUs
        dist.all_to_all_single(recv_tokens_flat, send_tokens_flat)
        dist.all_to_all_single(recv_expert_ids_flat, send_expert_ids_flat)
        dist.all_to_all_single(recv_weights_flat, send_weights_flat)
        dist.all_to_all_single(recv_mask_flat, send_mask_flat)

        # Reshape received data
        num_received_tokens = total_token_elements // self.n_embd
        recv_tokens = recv_tokens_flat.view(num_received_tokens, self.n_embd)
        recv_expert_ids = recv_expert_ids_flat.view(num_received_tokens, self.k)
        recv_weights = recv_weights_flat.view(num_received_tokens, self.k)
        recv_mask = recv_mask_flat.view(num_received_tokens, self.k)

        return recv_tokens, recv_expert_ids, recv_weights, recv_mask   
        


    def _process_local_experts(self, tokens, expert_ids, weights, mask):
        """
        Process tokens through local experts using expert assignments
        """
        device = tokens.device
        num_received_tokens = tokens.shape[0]

        print(f"[DEBUG] Rank {self.rank}: Processing {num_received_tokens} tokens through local experts")
        
        print(f"[DEBUG] Rank {self.rank}: Input shapes - tokens: {tokens.shape}, expert_ids: {expert_ids.shape}, weights: {weights.shape}, mask: {mask.shape}")

        print(f'\n[DEBUG] mask: {mask}\n')


        if num_received_tokens == 0:
            print(f"[DEBUG] Rank {self.rank}: No tokens to process, returning empty tensors")
            empty_output = torch.empty(0, self.n_embd, device=device, dtype=tokens.dtype)
            empty_counts = torch.zeros(self.experts_per_gpu, dtype=torch.int, device=device)
            return empty_output, empty_counts
        
        # create tensor to hold the output of locas experts.
        output = torch.zeros(num_received_tokens, self.n_embd, device=device, dtype=tokens.dtype) # (num_received_tokens, n_embd)

        # iterate over each token and process it through its assigned experts
        print(f"[DEBUG] Rank {self.rank}: Starting token-by-token processing...")
        for token_idx in range(num_received_tokens):
            
            if token_idx % 1000 == 0:  # Progress indicator for large batches
                print(f"[DEBUG] Rank {self.rank}: Processing token {token_idx}/{num_received_tokens}")

            token = tokens[token_idx]  # (n_embd,)
            print(f'[DEBUG] token shape: {token.shape}')
            
            token_expert_ids = expert_ids[token_idx] # (k,)
            print(f'[DEBUG] token_expert_ids {token_expert_ids.shape} {token_expert_ids}')
            
            token_weights = weights[token_idx] # (k,)
            print(f'[DEBUG] token_weights: {token_weights.shape} {token_weights}')
            token_expert_id_mask = mask[token_idx] # (k,)
            print(f'[DEBUG] token_expert_id_mask: {token_expert_id_mask.shape}\n{token_expert_id_mask}\n')


            # create tensor to hold this token after processing by experts
            processed_token = torch.zeros_like(token)


            # iterate over each expert assigned to this token
            for k_idx in range(self.k):
                print(f'[DEBUG] iterate over top k experts. at expert: {k_idx} of {self.k}')
                if token_expert_id_mask[k_idx]:  # check if this expert is real and not padding
                    print('here')
                    expert_local_id = token_expert_ids[k_idx].item()
                    expert_weight = token_weights[k_idx]

                   

                    # process the token through the expert
                    expert_output = self.local_experts[expert_local_id](token.unsqueeze(0))  # (1, n_embd)
                    
                    # Add the Weighted sum to the processed token tensor
                    processed_token += expert_output.squeeze(0) * expert_weight  # (n_embd,)

                    # Increment the count of tokens processed by this expert
                    self.count_tokens_processed_by_each_expert[expert_local_id] += 1
                    
            
            # Store the processed token in the output tensor
            output[token_idx] = processed_token
            
        tokens_processed_by_each_expert = self.count_tokens_processed_by_each_expert

        return output, tokens_processed_by_each_expert
    
    def _communicate_results_back(self, processed_tokens, assignments):
        """
        Communicate processed tokens back to the original GPUs
        """
        device = processed_tokens.device

        # Create a dictionary to hold a reverse mapping to send processed tokens back to the original GPU
        send_back_assignments = {}
        token_idx = 0

        for gpu_rank in range(self.world_size):
            if gpu_rank in assignments:
                token_positions, _, _, _ = assignments[gpu_rank]
                num_tokens = len(token_positions)

                if num_tokens > 0:
                    tokens_to_send_back_to_this_gpu = processed_tokens[token_idx : token_idx + num_tokens]
                    send_back_assignments[gpu_rank] = (token_positions, tokens_to_send_back_to_this_gpu)
                    token_idx += num_tokens

        # Create send tensors for all_to_all back to original GPUs
        send_tensors_back = []
        send_counts_back = []

        for gpu_rank in range(self.world_size):
            if gpu_rank in send_back_assignments:
                _, tokens_to_send_back_to_this_gpu = send_back_assignments[gpu_rank]
                send_tensors_back.append(tokens_to_send_back_to_this_gpu)
                send_counts_back.append(tokens_to_send_back_to_this_gpu.numel())
            else:
                empty_tensor = torch.empty(0, self.n_embd, device=device, dtype=processed_tokens.dtype)
                send_tensors_back.append(empty_tensor)
                send_counts_back.append(0)

        # Fix: properly handle all_gather_object
        all_send_counts_back = [None] * self.world_size
        dist.all_gather_object(all_send_counts_back, send_counts_back)
        
        # all_send_counts_back is now a list of lists - flatten appropriately
        # all_send_counts_back[rank] gives the send_counts from that rank
        
        # Compute total number of elements to receive on this GPU
        total_recv_size = sum(counts[self.rank] for counts in all_send_counts_back)
        
        recv_tokens_back_flat = torch.empty(total_recv_size, device=device, dtype=processed_tokens.dtype)

        # Flatten the send tensors for all_to_all_single communication
        send_tensors_back_flat = torch.cat([t.flatten() for t in send_tensors_back])

        # All-to-all communication back
        dist.all_to_all_single(recv_tokens_back_flat, send_tensors_back_flat)

        # Reshape the received tokens
        recv_tokens_back = recv_tokens_back_flat.view(-1, self.n_embd)

        return recv_tokens_back
    
    def _reassemble_sequence(self, x_flat, recv_tokens_back, original_assignments):
        """
        reassmeble the processed tokens back into the original sequence order
        """
        device = x_flat.device

        # create a tensor to hold reassebled sequence
        reassembled_sequence = torch.empty_like(x_flat)  # (batch_size * seq_len, n_embd)

        # Recall that recv_tokens_back contains processed tokens in GPU order, not the original sequence order. So we loop over each GPU, extract the original token positions from the GPU that we had sent tokens to for processing, and place the processed tokens back in their original positions in the reassembled_sequence tensor
        token_idx_back = 0 # track positions in recv_tokens_back.
        for gpu_rank in range(self.world_size):
            if gpu_rank in original_assignments:
                token_positions, _, _,_= original_assignments[gpu_rank] # get the token positions that were assigned to this gpu_rank for processing. These are the original positions in the sequence where these tokens were located before processing.
                num_tokens = len(token_positions)

                if num_tokens > 0:
                    # Get the processed tokens for these positions
                    processed_tokens_for_this_gpu = recv_tokens_back[token_idx_back:token_idx_back + num_tokens]  # (num_tokens, n_embd)

                    # use the token_positions to index into reassembled_sequence and place the processed tokens back in their original positions. reassembled_sequence[token_positions] = processed_tokens uses PyTorch's advanced indexing to scatter tokens to their correct positions in one operation
                    reassembled_sequence[token_positions] = processed_tokens_for_this_gpu
                    
                    token_idx_back += num_tokens # update the token index for the next GPU
        
        return reassembled_sequence
    
    def forward(self, x):
        """
        Complete MoE layer
        """
        batch_size, seq_len, _ = x.shape  # (batch_size, seq_len, n_embd)
        # x is the input tensor of shape (batch_size, seq_len, n_embd)
        device = x.device

        # Flatten the input tensor to (batch_size * seq_len, n_embd) for processing
        x_flat = x.view(-1, self.n_embd)

        # DEBUG: Check input consistency across ranks
        if dist.is_initialized():
            input_shape_tensor = torch.tensor([x_flat.shape[0], x_flat.shape[1]], device=device)
            all_shapes = [torch.zeros_like(input_shape_tensor) for _ in range(self.world_size)]
            dist.all_gather(all_shapes, input_shape_tensor)
            if self.rank == 0:
                print(f"[DEBUG] Input shapes across ranks: {[tuple(s.tolist()) for s in all_shapes]}")

        # Step 1: Get top-k expert assignments, weights, and load balance loss
        print(f"[DEBUG] Rank {self.rank}: Getting gate assignments...")
        top_k_gated_weights, top_k_ids_global, load_balance_loss = self.gate(x_flat)
        print(f"[DEBUG] Rank {self.rank}: Gate assignments complete, top_k_ids shape: {top_k_ids_global.shape}")

        # Step 2: Get expert assignments with padding
        print(f"[DEBUG] Rank {self.rank}: Getting expert assignments...")
        assignments = self._get_expert_assignments_with_padding(top_k_ids_global, top_k_gated_weights)
        
        # DEBUG: Print assignment sizes for each rank
        for gpu_rank in range(self.world_size):
            if gpu_rank in assignments:
                token_positions, _, _, _ = assignments[gpu_rank]
                print(f"[DEBUG] Rank {self.rank}: Sending {len(token_positions)} tokens to GPU {gpu_rank}")
            else:
                print(f"[DEBUG] Rank {self.rank}: Sending 0 tokens to GPU {gpu_rank}")

        # Step 3: Communicate tokens to appropriate GPUs
        print(f"[DEBUG] Rank {self.rank}: Starting token communication...")
        try:
            recv_tokens, recv_expert_ids, recv_weights, recv_mask = self._communicate_tokens(x_flat, assignments)
            print(f"[DEBUG] Rank {self.rank}: Token communication complete, received {recv_tokens.shape[0]} tokens")
        except Exception as e:
            print(f"[ERROR] Rank {self.rank}: Communication failed: {e}")
            raise

        # Step 4: Process tokens through local experts
        processed_tokens, count_tokens_processed_by_each_expert = self._process_local_experts(recv_tokens, recv_expert_ids, recv_weights, recv_mask)

        # Step 5: Communicate processed tokens back to original GPUs
        recv_tokens_back = self._communicate_results_back(processed_tokens, assignments)

        # Step 6: Reassemble the processed tokens back into the original sequence order
        reassembled_sequence_flat = self._reassemble_sequence(x_flat, recv_tokens_back, assignments)

        # Reshape the reassembled sequence back to (batch_size, seq_len, n_embd)
        reassembled_sequence = reassembled_sequence_flat.view(batch_size, seq_len, self.n_embd)

       
        
        return reassembled_sequence, load_balance_loss, count_tokens_processed_by_each_expert
        
        
# %%

class Block(nn.Module):
    """
    Create a full transformer block
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayerParallel(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_out, load_balance_loss, count_tokens_processed_by_each_expert = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, load_balance_loss, count_tokens_processed_by_each_expert
    
#%%
class CreateMoEParalell(nn.Module):
    """
    create the full model
    """    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classier projects from embedding dimension to vocab_size

        # weight tying design. the idea is to tie the weights of the input and output embeddings so that they are the same. This is done to reduce the number of parameters in the model and to improve generalization. 
        self.transformer.wte.weight = self.lm_head.weight

        # initialization
        self.apply(self._init_weights)

    #weight initialization. Mostly from GPT suggestions
    def _init_weights(self, module):
    # Standard GPT-style init
        std = 0.02

        if isinstance(module, nn.Linear):
            # ✅ Detect if this linear is inside an MoE expert
            if hasattr(module, "_am_expert"):
                # Experts get smaller init for stability
                std = 0.01
            
            # ✅ GPT residual scaling (e.g. output projection)
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            
            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None ):
        
        # idx is the input sequence of token ids
        B, T = idx.shape

        # this checks if the input sequence is longer than the block size
        assert T <= self.config.seq_len, f"Cannot forward sequence of length {T}, sequence length is only {self.config.seq_len}"

        # this creates the embedding table for the token ids.
        token_embd = self.transformer.wte(idx) # (B, T, n_embd)

        # apply the transformer blocks. each block applies layer norm, self-attention, residual connection, layer norm, MoE layer, residual connection
        x = token_embd
        load_balance_losses = []
        for block in self.transformer.h:
            x, load_balance_loss, count_tokens_processed_by_each_expert = block(x)
            load_balance_losses.append(load_balance_loss)

        # apply layer norm to the output of the last transformer block
        x = self.transformer.ln_f(x)

        # apply the final linear layer to get the logits for the next token prediction
        logits = self.lm_head(x) # (B, T, vocab_size)

        # if targets are provided, calculate the loss
        total_load_balance_loss = sum(load_balance_losses) / len(load_balance_losses)
        total_loss = None
        if targets is not None:
            # Pytorch's cross-entropy loss expects the logits to be of shape (B*T, vocab_size) and the targets to be of shape (B*T). So we need to reshape the logits and targets to match this shape.
            # reshape the logits: (B, T, vocab_size) -> (B*T, vocab_size) to match the shape of the targets: (B, T) -> (B*T) and then calculate the cross-entropy loss
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            # Note that load balance loss was already scaled in the TopKGateParallel module
            total_loss = main_loss + (total_load_balance_loss)
        
            return logits, total_loss, count_tokens_processed_by_each_expert
        
        return logits, total_loss, count_tokens_processed_by_each_expert