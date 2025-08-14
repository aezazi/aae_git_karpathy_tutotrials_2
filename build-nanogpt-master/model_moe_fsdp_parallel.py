#%%
#utility file to create model
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
    
    def forward(self, x):
        # x has shape (batch_size, sequence_length, embedding dimension) and is the output of the multi-head attention layer. 
        batch_size, seq_len, _ = x.shape
        # In each batch, there is a logit for each token in the sequence and for each expert. 
        logits = self.gate_linear(x) # (batch_size, seq_len, num_experts)

        # compute load balancing loss using clean logits before noise and topk
        gate_weights = F.softmax(logits, dim=-1)
        gate_weights_flat = gate_weights.view(batch_size*seq_len, -1)
        gate_weights_mean= gate_weights_flat.mean(0)  # (num_experts,) the mean of gate_weights over all tokens
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

        # local expert id range for this gou
        self.local_expert_start, self.local_expert_end = self.gpu_expert_ranges[self.rank]
    

        print(f"Rank {self.rank}: Managing experts {self.local_expert_start} to {self.local_expert_end-1}")

        # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each expert. the logits will have shape (batch_size, seq_len, num_experts) 
        self.gate = TopKGateParallel(config)

        # create local experts for this GPU
        self.local_experts = nn.ModuleList([
            ExpertMoESwiglu(config) for _ in range(self.experts_per_gpu)
        ])

        # NOTE: What is all-to-all communication: In a MoE setup, the topk gate will assign tokens being processed on this gpu to anyone of the num_experts. The experts assgined to a token may or may not be on this gpu. So we have to identify which gpu is hosting the expert to which a token is assgined, send the token to that gpu/expert for processing  and then receive the result back to rejoin the batch-sequence on this gpu.  
        
        # create communication buffers for all to all communication.
        self.send_buffer = None
        self.receive_buffer = None

    def _get_expert_assignments_with_padding(self, top_k_ids_global, gate_weights_global,config):
        """
        determine which tokens are assgined to which gpu by the topk gate.
        returns a dictionary that maps gpu global rank --> (token_indices, expert_local_id)
        """
        assignments = {}
    
        # flatten topk expert ids to shape (B*T, k) tensor for easier processing. 
        top_k_ids_flat = top_k_ids_global.view((self.batch_size*self.seq_len), -1)

        # Note: gate_weights_global is already of shape (batch_size*seq_len, num_experts) so no need to flatten.

        for gpu_rank in range(self.world_size):
            # get the start and end expert ids assigned to each gpu
            gpu_expert_start, gpu_expert_end = self.gpu_expert_ranges[gpu_rank]

            # a mask to filter all tokens in the sequence with an assigned topk expert id on this gpu_rank. The mask is a 1D tensor that will return TRUE for any row (token) that contains an expert in the range of experts assigned to this gpu_rank and FALSE otherwise. Note that for tensors, we need to use bitwise comparison operators. So use "&" instead of "and"
            expert_mask = ((top_k_ids_flat>=gpu_expert_start) & (top_k_ids_flat < gpu_expert_end)).any(dim=-1) # (batch_size*seq_len,)

            if expert_mask.any():
                # We extract the positions in expert_mask with value TRUE. These positions correspond to the token positions (indicies) in the sequence that were assigned to experts on this gpu_rank. note that using torch.where(condition) with just one argument returns a tuple of 1-D tensors, where each tensor represents the indices of the elements in the input condition that evaluate to True along each dimension of the input tensor. The first element in the tuple corresponds to the row indices and the second element corresponds to the column indices. We need the row indices (element 0) which correspond to the token positions in the sequence.
                token_positions = torch.where(expert_mask)[0]

                # For each token which has one or both of its assigned experts on this gpu_rank, get its local expert assignments. The assginments may have different number of experts between 1 and k experts. all to all communication expects all tensors to be of the same size and dtype. So we have to pad all expert assignments to the same length (k). We also need to create a mask that we will eventually use to filter the padding.
                token_expert_local_id_assignments_padded = []
                token_weights_padded =[]
                
                for token_idx in token_positions:
                    # this returns the the topk (k=2 in this case) experts to which this token is assigned. One or both of the experts might be on this gpu_rank.
                    token_experts = top_k_ids_flat[token_idx]


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
    
    def _communicate_tokens(self, x_flat, assignments, top_k_gated_weights, top_k_indices_global, batch_size, seq_len):
        pass
    """
    Perform all-to-all communication to send tokens to appropriate GPUs
    """
            
                
    def _extract_local_gating_weights(self, top_k_gated_weights, top_k_indices_global, token_positions, flattened_experts, expert_counts):
        """Extract gating weights for the tokens and experts being processed locally"""
        # This needs to be implemented to extract the correct gating weights
        # corresponding to the flattened expert assignments
        pass

    def _process_local_experts(self, tokens, flattened_experts, expert_counts, gating_weights):
        """
        Process tokens through local experts using flattened expert assignments
        """
        pass




        