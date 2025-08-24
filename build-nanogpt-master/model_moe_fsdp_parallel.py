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

    def _compute_load_balance_loss(self, logits):
        # the code here was suggested by Claude
        """Compute load balancing loss to encourage uniform expert usage"""
        # get pre-noise gate_weights for load_balance_loss computation
        gate_weights = F.softmax(logits, dim=-1)
        
        gate_weights_mean= gate_weights.mean(0)  # (num_experts,) the mean of gate_weights over all tokens
        
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
        
        # project x_flat to (token_count, num_experts)
        logits = self.gate_linear(x_flat) # (batch_size*seq_len, num_experts)

        # The global noise to be added to the logits of each expert. This is a random noise tensor of the same shape as the logits multiplied by noisy_std to create desired level of variance
        noise = torch.randn_like(logits) * self.noisy_std # (batch_size*seq_len, num_experts)

        # Per-expert noise scaling using self.noise_weights. 
        noise = noise * self.noise_weight

        # Add the noise to the logits. 
        logits_noisy = logits + noise  # (batch_size*seq_len, num_experts)

         # Get the top-k logits and their corresponding indices. The pytorch top_k method returns the top-k values and their indices along the last dimension (num_experts). In each batch, for each token in the sequence, it selects the top-k logits and their indices from the logits produced by each expert. Note that the top_k ids are global and not necessarily on this gpu.
        top_k_logits_noisy, top_k_ids_global_noisy = logits_noisy.topk(self.k, dim=-1)  # (batch_size* seq_len, top_k) 

        # We want sparse matrices. We achieve this by keeping the top-k logits for each token and filling the rest with a value that represents "no contribution" (like negative infinity).  This is done to ensure that the softmax function will ignore these values when computing the weights for each expert. So only the values produced by the top-k experts will contribute to the final output. We implement this by first creating a tensor of the same shape as the logits filled with negative infinity, and then using the scatter function to fill in the top-k logits at the indices of the top-k experts. Note that in this context, softmax is being used to compute "weights" for each expert not probabilities as for multiclass classification. Its a subttle difference but I think important to note.
        
        #full_like clones a tensor and fills it with a specified value (like infinity).
        zeros = torch.full_like(logits_noisy, float('-inf')) 

        # scatter fills the zeros tensor with the top_k_logits at the indices of the top_k_ids_global_noisy along the last dimension (-1). This creates a sparse matrix where only the top-k logits are kept and the rest are filled with negative infinity.
        sparse_logits_noisy = zeros.scatter(-1, top_k_ids_global_noisy, top_k_logits_noisy)

        # Note top_k_gated_weights has shape #(B*seq_len, num_experts). The top_k experts will have weights that sum to 1, the other experts will have weights-inf
        top_k_gated_weights = F.softmax(sparse_logits_noisy, dim=-1) 

        # compute load balance loss using pre-noise logits
        load_balance_loss = self._compute_load_balance_loss(logits)

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
    
    def _get_expert_assignments_with_padding(self, top_k_ids_global, top_k_gate_weights_global):
        """
        determine which tokens are assgined to which gpu by the topk gate.
        returns a dictionary that maps gpu global rank --> (token_indices, expert_local_id)
        """
        
        assignments = {}

        for gpu_rank in range(self.world_size):
            # get the start and end expert ids assigned to each gpu
            gpu_expert_start, gpu_expert_end = self.gpu_expert_ranges[gpu_rank]

            # a mask to filter all tokens in the sequence with an assigned topk expert id on this gpu_rank. The mask is a 1D tensor that will return TRUE for any row (token) that contains an expert in the range of experts assigned to this gpu_rank and FALSE otherwise. Note that for tensors, we need to use bitwise comparison operators. So use "&" instead of "and". Note that this filters out tokens that don't have any of their k=2  assigned experts on this gpu. If a token has just one of it's assigned experts on this gpu, it passes through this filter. In such a case, We still need to filter out the expert that is not on this gpu. This is done on a token by token basis below
            expert_mask = ((top_k_ids_global>=gpu_expert_start) & (top_k_ids_global < gpu_expert_end)).any(dim=-1) # (batch_size*seq_len,)

            if expert_mask.any():
                # We extract the positions in expert_mask with value TRUE. These positions correspond to the token positions (indicies) in the sequence were tokens have at least one assigned experts on this gpu_rank. note that using torch.where(condition) with just one argument returns a tuple of 1-D tensors, where each tensor represents the indices of the elements in the input condition that evaluate to True along each dimension of the input tensor. The first element in the tuple corresponds to the row indices and the second element corresponds to the column indices. We need the row indices (element 0) which correspond to the token positions (rows) in the sequence.
                token_positions = torch.where(expert_mask)[0]

                # For each token which has one or both of its assigned experts on this gpu_rank, get its local expert assignments. The assginments may have different number of experts between 1 and k experts. all to all communication expects all tensors to be of the same size and dtype. So we have to pad all expert assignments to the same length (k). 
                top_k_expert_local_id_assignments_padded = []
                top_k_weights_padded =[]
                
                for token_idx in token_positions:
                    # this returns the the topk (k=2 in this case) experts to which this token is assigned. One or both of the experts might be on this gpu_rank.
                    token_experts = top_k_ids_global[token_idx]


                    # create a mask to filter token experts on this gpu_rank. since one or both of the experts to which this token was assigned may be on this gpu_rank, we need to filter for only the experts on this gpu_rank. 
                    gpu_token_expert_mask = ((token_experts >= gpu_expert_start) & 
                                      (token_experts < gpu_expert_end))
                    
                    # apply the mask to get only token experts on this gpu_rank
                    token_experts_on_this_gpu = token_experts[gpu_token_expert_mask]

                    # use the token_experts_on_this_gpu to select from top_k_gate_weights_global, the weights for this token with an expert on this gpu_rank.
                    top_k_weights_on_this_gpu = top_k_gate_weights_global[token_idx][token_experts_on_this_gpu]

                    # the token expert ids we obtained above are global id range (0 - num_experts). we need to convert these to local ids since on each gpu the experts are indexed 0 to experts_per_gpu-1.
                    token_experts_local_ids = token_experts_on_this_gpu - gpu_expert_start

                    # get the number of experts on this gpu to which this token was assigned
                    token_num_experts = len(token_experts_local_ids) 
                    
                    if token_num_experts < self.k:
                        # if the number of experts is less than k, pad to length k with -1 to indicate dummy expert
                        top_k_expert_local_id_assignments_padded.append(F.pad(token_experts_local_ids, (0, self.k - token_num_experts), value=-1))

                        top_k_weights_padded.append(F.pad(top_k_weights_on_this_gpu, (0, self.k - token_num_experts), value=0.0))

                    else:
                        # if the number of experts is already k, no padding needed
                         top_k_expert_local_id_assignments_padded.append(token_experts_local_ids)  
                         
                         top_k_weights_padded.append(top_k_weights_on_this_gpu)

                    # by index, each element in top_k_expert_local_id_assignments_padded is the local id(s) of experts to which tokens in token_positions was assigned. So token at token_positions[3] was assigned to expert(s) at top_k_expert_local_id_assignments_padded[3]
                    

                # stack the padded experts ids into a tensor. This will have shape (num_tokens, k) where num_tokens is the number of tokens assigned to this gpu_rank and k is the number of experts per token.
                top_k_expert_local_id_assignments_padded = torch.stack(top_k_expert_local_id_assignments_padded)

                # stack the padded weights into a tensor same as done aove for top_k_expert_local_id_assignments_padded
                top_k_weights_padded = torch.stack(top_k_weights_padded)

                # create a mask to filter the padding. the mask will have shape (num_tokens, k) where num_tokens is the number of tokens assigned to this gpu_rank and k is the number of experts per token. The mask will be True for the experts that were assigned to the token and False for the padding dummy.
                # token_expert_assignment_mask = (top_k_expert_local_id_assignments_padded >= 0)  # (num_tokens, k)
                
                # finally, package token_positions, their top_k_expert_local_id_assignments_padded, and token_expert_assignment_mask in a tuple and associate the tuple with this gpu_rank on which the experts reside. for example, assignments[1] carries the token positions and the experts to which the tokens were assigned and mask that can be processed by gpu rank=1
                assignments[gpu_rank] = (token_positions, top_k_expert_local_id_assignments_padded,
                top_k_weights_padded)
        
        return assignments
    
   
    def _communicate_tokens(self, x_flat, assignments):
        """
        Perform all-to-all communication to send tokens to appropriate GPUs
        """
        print('-'*70)
        print(f'[DEBUG] Rank {self.rank} entered _communicate_tokens')
        device = x_flat.device

        # prepare send data for each gpu. store the the token_positions, top_k_expert_local_id_assignments_padded, top_k_weights_padded, token_expert_assignment_mask tensors to each gpu
        # Buckets per rank
        send_tokens_list = []
        send_expert_list = []
        send_weights_list = []
        # send_mask_list = []
        send_counts = []
        
        # send_counts carries how many rows (tokens) will be sent by this gpu to all gpus (including itself). The list will have len = world_size. Each element will be an integer corresponding to how many rows (tokens) this gpu will send to all gpus ( including itself). Each index position corresponds to gpu rank in order. 
        send_counts = []
        
        for gpu_rank in range(self.world_size):
            if gpu_rank in assignments:
                token_positions, top_k_expert_local_id_assignments_padded, top_k_weights_padded = assignments[gpu_rank]
                
                print(f'[DEBUG] Rank {self.rank} shape top_k_expert_local_id_assignments_padded : {top_k_expert_local_id_assignments_padded.shape}')

                print(f'[DEBUG] Rank {self.rank} top_k_weights_padded: {top_k_weights_padded.shape}')

                # extract the tokens with at least one expert on this gpu_rank. 
                send_tokens = x_flat[token_positions] # (num_tokens, n_embd)
                send_tokens_list.append(send_tokens)
                # print(f'[DEBUG] Rank {self.rank} send_tokens shape: {send_tokens.shape}')
                
                # Note that for expert_ids, weights, and mask, the assginments method already filtered out tokens with no expert assignments on this gpu.
                send_expert_ids = top_k_expert_local_id_assignments_padded
                send_expert_list.append(send_expert_ids)
                # print(f'[DEBUG] Rank {self.rank} send_expert_ids shape: {send_expert_ids.shape}')

                # send_weights = top_k_weights_padded
                send_weights_list.append(top_k_weights_padded)
                # print(f'[DEBUG] Rank {self.rank} send_weights shape: {send_weights.shape}')

                send_counts.append(send_tokens.shape[0])
                # print(f'[DEBUG] Rank {self.rank} send counts in loop: {send_counts}')
                
            else:
                # if no tokens assigned to this gpu_rank, send dummy tensors with 0 rows
                dummy_tokens = torch.empty((0, self.n_embd), device=device, dtype=x_flat.dtype)
                send_tokens_list.append(dummy_tokens)

                dummy_expert_ids = torch.empty((0,self.k), device=device, dtype=torch.long)
                send_expert_list.append(dummy_expert_ids)

                dummy_weights = torch.empty((0, self.k), device=device, dtype=x_flat.dtype)
                send_weights_list.append(dummy_weights)

                send_counts.append(0)

    
        print(f"[DEBUG] Rank {self.rank} send counts after looping gpus: = {send_counts}, sum = {sum(send_counts)}, tensor_rows = {send_tokens.size(0)}")
        
        # create send tensors
        send_tokens_tensor = torch.cat(send_tokens_list, dim=0) if sum(send_counts) >0 else torch.empty((0, self.n_embd), device=device)

        send_expert_ids_tensor = torch.cat(send_expert_list, dim=0) if sum(send_counts) >0 else torch.empty((0, self.k), device=device)

        send_weights_tensor = torch.cat(send_weights_list, dim=0) if sum(send_counts) >0 else torch.empty((0, self.k), device=device)

        assert len(send_counts) == dist.get_world_size()
        assert sum(send_counts) == send_tokens_tensor.shape[0], "Mismatch between send_counts and tokens!"
        assert all(c >= 0 for c in send_counts), "send_counts has negative entries!"


        # The first step in all-to-all communication is to communicate how many rows each rank will be receiving. This  allows us to create "receiving" tensors of the correct shape that will be populated by all-to-all. All-to-all expects a single contiguous tensor for send and receive. This single tensor is 'split' along the first dimension of the tensor between gpus. In our case, the first dimension is always  number of tokens (n_tokens, n_embd) or (n_tokens).  The input_split_sizes tensor tells this  gpu how many rows it will be receiving from other gpus. The output_split_sizes_tensor tells Pytorch how to split the tensors sent by this gpu among other gpus. recall that send_counts carries how many rows (tokens) will be sent by this gpu to all gpus (including itself). So the input_split_sizes_tensor will be created using the send_counts list. Both input_split_sizes_tensor and output_split_sizes_tensor will have shape (world_size,) p
        

        input_split_sizes_tensor = torch.tensor(send_counts, device=device, dtype=torch.int) # num rows this gpu will be sending to each gpu (world_size,)
        print(f'[DEBUG] Rank {self.rank} input_split_sizes_tensor: {input_split_sizes_tensor}\n')
        
        output_split_sizes_tensor = torch.empty_like(input_split_sizes_tensor) # num rows this gpu will be receiving from each gpu (world_size,). It will be populated after dist_all_to_all single
        
        # communicate
        print(f'[DEBUG] Rank {self.rank} initiating dist.all_to_all_single(recv_counts, send_counts)\n')
        dist.all_to_all_single(output_split_sizes_tensor, input_split_sizes_tensor)
        
        # after the all-to-all communication, output_split_sizes_tensor is populated. convert to a list
        recv_counts = output_split_sizes_tensor.tolist()
        
        print(f'[DEBUG] Rank {self.rank} recv_counts: {recv_counts}')
    

        # this is the sum of the number of token (rows) that all gpus(including itself) will be sending to this gpu. N_recv will be the same for the tokens (N_recv, n_embd), expert_ids (N_recv, k), weights (N_recv, k), and mask tokens (N_recv, k)
        N_recv = sum(output_split_sizes_tensor)
        print(f'[DEBUG] Rank {self.rank} N_recv: {N_recv}')

        
        # Create tensors to receive what all gpus will communicate to this gpu
        # Tokens: each row is a token embedding. Tokens (rows) are in the order of gpu rank 
        recv_tokens_tensor = torch.empty((N_recv, self.n_embd), dtype=x_flat.dtype, device=device)

        # Expert IDs: each row has k integers (which experts this token is routed to)
        recv_expert_ids_tensor = torch.empty((N_recv, self.k), dtype=torch.long, device=device)

        # Weights: each row has k floats (routing probabilities)
        recv_weights_tensor = torch.empty((N_recv, self.k), dtype=torch.float32, device=device)

        # Mask: each row has k 0/1 flags (in case of padding or variable k)
        # recv_mask_tensor = torch.empty((N_recv, self.k), dtype=torch.bool, device=device)

        # all-to-all communication to send tokens to appropriate GPUs
        dist.all_to_all_single(recv_tokens_tensor, send_tokens_tensor, output_split_sizes=recv_counts, input_split_sizes=send_counts)
        
        dist.all_to_all_single(recv_expert_ids_tensor, send_expert_ids_tensor, output_split_sizes=recv_counts, input_split_sizes=send_counts)
        
        dist.all_to_all_single(recv_weights_tensor, send_weights_tensor, output_split_sizes=recv_counts, input_split_sizes=send_counts)

        print(f"\n[DEBUG] Rank {self.rank}: Token communication complete, received {recv_tokens_tensor.shape[0]} tokens\n")
            
        # print(f"\n[DEBUG] recv_tokens sample\n: {recv_tokens_tensor[0:10,0:7]}")


        return recv_tokens_tensor, recv_expert_ids_tensor, recv_weights_tensor, recv_counts
        

    def _process_local_experts(self, tokens, expert_ids, top_k_weights):
        """
        Process tokens through local experts using expert assignments
        """
        print('-'*70,'\n')
        print(f'[DEBUG] Rank {self.rank} entered _process_local_experts\n')

        device = tokens.device
        num_received_tokens = tokens.shape[0]

        print(f"[DEBUG] Rank {self.rank}: Processing {num_received_tokens} tokens through local experts\n")
        
        print(f"[DEBUG] Rank {self.rank}: Input shapes - tokens: {tokens.shape}, expert_ids: {expert_ids.shape}, top_k_weights: {top_k_weights.shape}\n")


        if num_received_tokens == 0:
            print(f"[DEBUG] Rank {self.rank}: No tokens to process, returning empty tensors")
            empty_output = torch.empty(0, self.n_embd, device=device, dtype=tokens.dtype)
            empty_counts = torch.zeros(self.experts_per_gpu, dtype=torch.int, device=device)
            return empty_output, empty_counts
        
        # create tensor to hold the output of locas experts.
        output = torch.zeros(num_received_tokens, self.n_embd, device=device, dtype=tokens.dtype) # (num_received_tokens, n_embd)

        # iterate over each token and process it through its assigned experts
        print(f"[DEBUG] Rank {self.rank}: Starting token-by-token processing...\n")
        for token_idx in range(num_received_tokens):
            
            if token_idx % 5000 == 0:  # Progress indicator for large batches
                print(f"[DEBUG] Rank {self.rank}: Processing token {token_idx}/{num_received_tokens}")

            token = tokens[token_idx]  # (n_embd,)
            # print(f'[DEBUG] token shape: {token.shape}')
            
            token_expert_ids = expert_ids[token_idx] # (k,)
            # print(f'[DEBUG] token_expert_ids {token_expert_ids.shape} {token_expert_ids}')
            
            token_weights = top_k_weights[token_idx] # (k,)
            
            # if token_idx % 5000 == 0: print(f'[DEBUG] token_weights: {token_weights}\n')

            # create tensor to hold this token after processing by experts
            processed_token = torch.zeros_like(token)


            # iterate over each expert assigned to this token
            for k_idx in range(self.k):

                # k_idx is the local id of topk experts. check if this expert is real and not padding. if the expert assginment k_idx != -1, it's real.
                if k_idx != -1:
                    expert_local_id = token_expert_ids[k_idx].item()
                    expert_weight = token_weights[k_idx]
                   
                    # process the token through the expert
                    expert_output = self.local_experts[expert_local_id](token.unsqueeze(0))  # (1, n_embd)
                    
                    # Add the Weighted sum to the processed token tensor. The weighted sum
                    processed_token += expert_output.squeeze(0) * expert_weight  # (n_embd,)

                    # Increment the count of tokens processed by this expert
                    self.count_tokens_processed_by_each_expert[expert_local_id] += 1
                    
            
            # Store the processed token in the output tensor
            output[token_idx] = processed_token
            
        tokens_processed_by_each_expert = self.count_tokens_processed_by_each_expert
        print(f"\n[DEBUG] Rank {self.rank}: num tokens processed by this rank: {output.shape}\n")

        return output, tokens_processed_by_each_expert
    
    def _communicate_results_back(self, processed_tokens, recv_counts_forward):
        """
        Communicate processed tokens back to the original GPUs using reverse all-to-all
        
        Args:
            processed_tokens: Tensor of shape (N_recv, n_embd) - tokens processed by this GPU's experts  
            recv_counts_forward: List of ints - how many tokens this GPU received from each GPU in forward pass
        
        Returns:
            recv_tokens_back: Tensor containing processed tokens from all GPUs in GPU rank order
        """
        print('-'*70)
        print(f'[DEBUG] Rank {self.rank} entered _communicate_results_back')
        print(f'\n[DEBUG] Rank {self.rank} processed {processed_tokens.shape[0]} tokens total')
        print(f'[DEBUG] Rank {self.rank} forward recv_counts were: {recv_counts_forward}\n')
        
        device = processed_tokens.device

        # Step 1: Prepare send data - split processed tokens back according to forward receive counts. Whatever tokens this gpu received and proceesed must be sent back to the gpus that sent the tokens
        # The processed_tokens are in the same order as we received them (GPU rank order)
        send_counts_back = recv_counts_forward.copy()  # Send back exactly what we received
        send_tokens_list_back = []

        token_idx = 0  # Track position in processed_tokens
    
        for gpu_rank in range(self.world_size):
            num_tokens = send_counts_back[gpu_rank]
            
            if num_tokens > 0:
                # Extract the processed tokens for this GPU
                tokens_for_gpu = processed_tokens[token_idx:token_idx + num_tokens]  # (num_tokens, n_embd)
                send_tokens_list_back.append(tokens_for_gpu)
                token_idx += num_tokens
                
                print(f'\n[DEBUG] Rank {self.rank} sending {num_tokens} processed tokens back to GPU {gpu_rank}\n')
            else:
                # No tokens to send back to this GPU
                dummy_tokens = torch.empty((0, self.n_embd), device=device, dtype=processed_tokens.dtype)
                send_tokens_list_back.append(dummy_tokens)

        # Create the send tensor by concatenating all tokens to be sent back
        send_tokens_back = torch.cat(send_tokens_list_back, dim=0) if sum(send_counts_back) > 0 else torch.empty((0, self.n_embd), device=device, dtype=processed_tokens.dtype)
        
        print(f'\n[DEBUG] Rank {self.rank} prepared {send_tokens_back.shape[0]} tokens to send back\n')
        
        # Step 2: Communicate send counts for the reverse direction
        # We need to tell all GPUs how many tokens we're sending back to them
        input_split_sizes_back = torch.tensor(send_counts_back, device=device, dtype=torch.int)
        output_split_sizes_back = torch.empty_like(input_split_sizes_back)
        
        print(f'\n[DEBUG] Rank {self.rank} communicating reverse send counts: {send_counts_back}\n')

        # All-to-all to exchange how many tokens each GPU will receive back
        dist.all_to_all_single(output_split_sizes_back, input_split_sizes_back)

        recv_counts_back = output_split_sizes_back.tolist()
        N_recv_back = sum(recv_counts_back)
        
        print(f'[DEBUG] Rank {self.rank} will receive {recv_counts_back} tokens back from all GPUs (total: {N_recv_back})')
        
        # Step 3: Create receive tensor and perform all-to-all communication
        recv_tokens_back = torch.empty((N_recv_back, self.n_embd), dtype=processed_tokens.dtype, device=device)
        
        # Perform the reverse all-to-all communication
        print(f'[DEBUG] Rank {self.rank} performing reverse all-to-all communication')
        dist.all_to_all_single(
            recv_tokens_back, 
            send_tokens_back, 
            output_split_sizes=recv_counts_back, 
            input_split_sizes=send_counts_back
        )
        
        print(f'[DEBUG] Rank {self.rank} received {recv_tokens_back.shape[0]} processed tokens back from all GPUs')
        
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
                token_positions, _, _ = original_assignments[gpu_rank] # get the token positions that were assigned to this gpu_rank for processing. These are the original positions in the sequence where these tokens were located before processing.
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
                print(f"[DEBUG] shape of data to each gpu: {[tuple(s.tolist()) for s in all_shapes]}")

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
                token_positions, _, _ = assignments[gpu_rank]
                print(f"[DEBUG] Rank {self.rank}: Sending {len(token_positions)} tokens to GPU {gpu_rank}")
            else:
                print(f"[DEBUG] Rank {self.rank}: Sending 0 tokens to GPU {gpu_rank}")

        # Step 3: Communicate tokens to appropriate GPUs
        recv_tokens, recv_expert_ids, recv_weights, recv_counts_forward = self._communicate_tokens(x_flat, assignments)

        
        # Step 4: Process tokens through local experts
        processed_tokens, count_tokens_processed_by_each_expert = self._process_local_experts(recv_tokens, recv_expert_ids, recv_weights)

        # Verify we processed exactly what we received
        expected_processed = sum(recv_counts_forward)
        actual_processed = processed_tokens.shape[0]
        print(f"\n[DEBUG] Rank {self.rank}: Expected to process {expected_processed} tokens, actually processed {actual_processed}\n")
        assert expected_processed == actual_processed, f"Token count mismatch! Expected {expected_processed}, got {actual_processed}"

        # DEBUG:  check tokens processed counter
        
        # Step 5: Communicate processed tokens back to original GPUs
        recv_tokens_back = self._communicate_results_back(processed_tokens, recv_counts_forward)

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