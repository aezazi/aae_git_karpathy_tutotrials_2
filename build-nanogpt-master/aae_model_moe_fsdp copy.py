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
        self.head_dim = config.n_embd/config.n_head
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
        x1 = x[:, :, :, : :2]
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
        
        # transpose back to (B, seq_len, n_heads*dim_heads) and combine heads. Note that the y vector returned by scaled_dot_product is not contiguous and therefor view cannot be applied to until we make it . For view() to work, the original tensor must be contiguous in memory. reshape() can work with both contiguous and non-contiguous tensors, automatically handling the necessary memory operations. 
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
class TopKMoEGate(nn.Module):
    def __init__(self, config, local_expert_ids):
        super().__init__()
        self.local_expert_ids = local_expert_ids
        self.num_local_experts = len(self.local_expert_ids)
        self.k = config.k
        self.seq_len = config.seq_len
    
        # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each local expert. The local experts are the experts assigned to the GPU running this instance. This is very important. I had  orignally made the mistake of doing nn.Linear(config.n_embd, self.num_experts, bias=False). This was computing logits over all num_experts, regardless of how many experts exist on each GPU.But the MoE layer (MoELayerSharded) only instantiates a subset of those experts per GPU (based on expert_id % world_size == local_rank), and applies only the ones available on that rank.

        # This results in:
        # 	• These tokens are ignored during the expert loop, i.e., dropped silently.
        # 	• Tokens being routed to experts that don’t exist locally.
        # 	• Thus, part of the model output remains zeros.
        # 	• Consequently, gradients don’t flow to many experts.
        # 	• Eventually, the gating network settles into a degenerate distribution over the few working experts.
        # 	• This is why my  loss was stalling at ~8.2 on multi-GPU — it’s only marginally better than random output.This doesn’t happen on single-GPU because all experts exist and are used, so the full distribution of the gating network is valid.
        self.gate_linear = nn.Linear(config.n_embd, self.num_local_experts, bias=False)

        # this is a learnable parameter that will be used to scale the noise added to the logits of each expert. The allows the noise added to each expert to be "customized" and dynamic. It adapts during training depending on the expert. It is initialized to zero and will be learned during training.
        self.noise_weight = nn.Parameter(torch.zeros(self.num_local_experts)) 

        # this is the standard deviation of the noise added to the logits of each expert. It is a hyperparameter that can be tuned. Note that unlike the noise_weight it is a global parameter, this is not a learnable parameter.
        self.noisy_std = 1.0
        
    # x has shape (batch_size, sequence_length, embedding dimension) and is the output of the multi-head attention layer. 
    def forward(self, x):
        
        # In each batch, there is a logit for each token in the sequence and for each expert. 
        logits = self.gate_linear(x) # (batch_size, seq_len, num_experts)

        # The global noise to be added to the logits of each expert. This is a random noise tensor of the same shape as the logits. 
        noise = torch.randn_like(logits) * self.noisy_std # (batch_size, seq_len, num_experts)

        # Per-expert noise scaling using self.weights. 
        noise = noise * self.noise_weight

        # Add the noise to the logits. 
        logits_noisy = logits + noise  # (batch_size, seq_len, num_experts)

        # Get the top-k logits and their corresponding indices. The pytorch top_k method returns the top-k values and their indices along the last dimension (num_experts). In each batch, for each token in the sequence, it selects the top-k logits and their indices from the logits produced by each expert.
        top_k_logits_noisy, top_k_indices_noisy = logits_noisy.topk(self.k, dim=-1)  # (batch_size, seq_len, top_k) 

        # We want sparse matrices. We achieve this by keeping the top-k logits for each token and filling the rest with a value that represents "no contribution" (like negative infinity).  This is done to ensure that the softmax function will ignore these values when computing the weights for each expert. So only the values produced by the top-k experts will contribute to the final output. We implement this by first creating a tensor of the same shape as the logits filled with negative infinity, and then using the scatter function to fill in the top-k logits at the indices of the top-k experts. Note that in this context, softmax is being used to compute "weights" for each expert not probabilities as for multiclass classification. Its a subttle difference but I think important to note.
        
        #full_like clones a tensor and fills it with a specified value (like infinity).
        mask = torch.full_like(logits_noisy, float('-inf')) 

        # scatter fills the zeros tensor with the top_k_logits at the indices of the top_k_indices along the last dimension (-1). This creates a sparse matrix where only the top-k logits are kept and the rest are filled with negative infinity.
        sparse_logits_noisy = mask.scatter(-1, top_k_indices_noisy, top_k_logits_noisy)
        top_k_gated_weights = F.softmax(sparse_logits_noisy, dim=-1)

        return top_k_gated_weights, top_k_indices_noisy


# %%
# this class implements the full MoE layer. It uses the TopKMoEGate class to compute the gated weights and the top-k indices, and then applies each expert to the input based on these indices. The output is a weighted sum of the expert outputs, where the weights are the gated weights.
class MoELayerSharded(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.sq_len = config.seq_len
        self.k = config.k
        # self.aux_loss_scale = config.aux_loss_scale
        # self.moe_scale = config.moe_scale

        # note that when on multiple gpus, FSDP or DDP launch identical processes on each gpu. So a separate instance of this module will be launched on each gpu. Each instance will have a different rank corresponding to the gpu it was launched on. dist.getrank() returns the rank of each process/gpu. In my case, all my gpus are on one on one machine, so local_rank = rank. 
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()


        # This is a clever algo GPT suggested for assigning experts to GPUs. It's called "round robin" assignment. It assigns only a subset of experts to the GPU running this instance of the module. If remainder of expert_idx/world_size == local_rank, create and add an expert to this GPU
        self.local_expert_ids = [e_idx for e_idx in range(self.num_experts) if e_idx % self.world_size == self.rank]

        self.num_local_experts = len(self.local_expert_ids)
                
        # Instantiate top_k gating. Note that I am passing the ids of the experts assigned to the GPU running this instance
        self.gate = TopKMoEGate(config, self.local_expert_ids)

        # create ModuleDict with an expert (instance of ExpertMoeSwiglu) for each index in local_experts_ids. 
        self.local_experts = nn.ModuleDict({str(i) : ExpertMoESwiglu(config) for i in self.local_expert_ids})
        
       
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Get the top_k gated weights and top-k indices from the gate for the local experts. Note that the top_k_indices are indexed based on the number of experts assigned to this gpu. So if this gpu has two experts assigned, each the top_k (k=2)indices for each token will be of the form [0,1] or [1,0]. If there are three experts on this gpu, each top_k index would be of the form [1,2] or [2,0]. These local expert indices have to mapped to the global expert indicies. So as an example, if we have 8 experts, and experts [3, 7] are assigned to this gpu, and the top_k_local_indices returned by the gate for a token are [1,0], this would make the  the top_k_global_indices [7,3] which means experts 7 and 3 were the top 2 picked by the gate for that token on this gpu.
        top_k_gated_weights, top_k_local_indices  = self.gate(x)

        # Put the  global id of the experts on this gpu into a tensor
        local_expert_id_tensor = torch.tensor(self.local_expert_ids, device=x.device)  # [num experts assigned to this gpu]
        # map the top_k_local_indices to the global id
        top_k_global_indices = local_expert_id_tensor[top_k_local_indices]  # [B, T, k]

        # tensor to hold the output from just the experts in this process
        y_partial_output = torch.zeros_like(x)
        # print(f'\ninput x shape: {x.shape} \n{x}\n')

        # flatten the input to (batch_size * seq_len, n_embd) for batch processing
        x_flat = x.view(batch_size*seq_len, -1) 
        # print(f'\ninput x_flat shape: {x_flat.shape} \n{x_flat}\n')

        # flatten the gated weights to (batch_size * seq_len, num_local_experts) for batch processing
        top_k_gated_weights_flat = top_k_gated_weights.view(batch_size*seq_len, self.num_local_experts)  

        # Iterate over each expert  assigned to this GPU and apply it to the input
        for expert_id_global in self.local_experts.keys():
            expert_id_global = int(expert_id_global)

            # Create a mask for the inputs where the current expert is in top-k. the mask will have shape (batch_size, seq_len). top_k_indices have shape (B, seq_len, top_k). Each row of each batch in top_k_indices has the indices of the top two experts for the token corresponding that row in the token sequence. The mask will return True (row wise) if expert i is in the top_k indices. S
            print(f'\nExpert {expert_id_global} with x_flat input shape {x_flat.shape}\n')
            print(f'top_k_indices shape: {top_k_global_indices.shape} \n{top_k_global_indices}\n')
            expert_mask = (top_k_global_indices == expert_id_global)
            print(f'expert_mask shape: {expert_mask.shape} \n{expert_mask}\n')

            # flatten the mask to match the shape of the flattened input x_flat. Note that the shape of flat_mask is a one dimensional (batch_size*seq_len). x_flat has shape (batch_size * seq_len, n_embd). each row in x_flat is a token in the sequence. 
            flat_mask = expert_mask.view(-1) # (batch_size * seq_len)
            # print(f'flat_mask shape: {flat_mask.shape} \n{flat_mask}\n')

            if flat_mask.any():
                # If the flat mask has any True values, Apply the expert to the inputs selected by the mask. x_flat[flat_mask] picks the rows(tokens) of x_flat where the mask is True. This allows us to activate the expert only for the tokens where the expert with expert_id_global is in the top_k indices.
                expert_input = x_flat[flat_mask] # (number of tokens where expert_id_global is in top_k, n_embd)
                # print(f'\nexpert_input shape: {expert_input.shape} \n{expert_input}\n')

                # apply expert i to the expert_input. Again, note that based on the mask described above, epxert i is applied only to the tokens where it is in the top_k indices.
                expert_output = self.experts[expert_id_global](expert_input) # (number of tokens where expert i is in top_k, n_embd)
                # print(f'expert_output shape: {expert_output.shape} \n{expert_output}\n')

                # Now we need to scale the expert output by the gated weights for the tokens where the expert is in the top_k indices. gated_weights_flat has shape (batch_size * seq_len, num_local_experts). We apply the same mask as we used to create expert_input to select all rows from gated_weights_flat where expert i is in the top_k indices, then we select the ith column. This returns the weighting for expert i that is to be applied to the tokens where expert i is in the top_k indices. This is the same as selecting all the non_zero values in the ith column of gated_weights_flat. We  then use unsqueeze(1) to add a dimension to create a column vector of shape (number of tokens where expert i is in top_k, 1). this allows  multiplication with the expert_output which has shape (number of tokens where expert i is in top_k, n_embd). 
                # print(f'gated_weights_flat shape: {gated_weights_flat.shape} \n{gated_weights_flat}\n')
                expert_weights = top_k_gated_weights_flat[flat_mask, expert_id_global].unsqueeze(1)  # (number of tokens where expert i is in top_k, 1)
                # print(f'expert_weights shape: {expert_weights.shape} \n{expert_weights}\n')

                # Scale the expert_output by expert_weights.
                expert_output_weighted = expert_output * expert_weights # (number of tokens where expert i is in top_k, n_embd)
                # print(f'expert_output_weighted shape: {expert_output_weighted.shape} \n{expert_output_weighted}\n')

                # the huggingface implementation uses .squeeze(1) to remove any singleton dimensions from  the expert_output_weighted tensor. Not sure why this is needed. I tried removing it and the shapes were still compatible and the result the same
                y_partial_output[expert_mask] += expert_output_weighted.squeeze(1) # (batch_size, seq_len, n_embd)
                # print(f'final_output shape after adding expert_output_weighted:{final_output.shape} \n{final_output}\n')

                ## just a test to see if .squeeze(1) is necessary
                # final_test = torch.zeros_like(x)
                # final_test[expert_mask] += expert_output_weighted
                # print(f'{torch.allclose(final_output, final_test)}')
        
        # each instance
        if self.world_size > 1:
            dist.all_reduce(y_partial_output, op=dist.ReduceOp.SUM)

        # compute auxiliary load balancing loss
        # first compute the average weight given to each expert by the top_k router. This is how much the router wanted to send to each expert.
        avg_weight_per_expert = top_k_gated_weights_flat.mean(0) # shape(num_experts)
        # print(f'\navg_weight_per_expert:\n {avg_weight_per_expert}\n')

        # Replaces elements != 0 with 1.0 and takes the mean over (batch*seq_len).result has shape(num_experts)
        avg_tokens_per_expert = torch.where(top_k_gated_weights_flat != 0, 1.0, 0).mean(0)

        # print(f'\navg_tokens_per_expert: \n {avg_tokens_per_expert}\n')


        # refer to literature on why this formula for the load balancing loss
        aux_loss = (avg_tokens_per_expert * avg_weight_per_expert).sum() * self.num_local_experts



        
#%%
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayerSharded(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss = self.moe(self.ln_2(x))  
        x = x + moe_out
        return x, aux_loss

# %%
class CreateMoESharded(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final classifier projects from embedding dimension to vocab_size

        # weight tying design. the idea is to tie the weights of the input and output embeddings so that they are the same. This is done to reduce the number of parameters in the model and to improve generalization. 
        self.transformer.wte.weight = self.lm_head.weight

        # initialization
        self.apply(self._init_weights)

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

    def forward(self, idx, targets=None):
        # idx is the input sequence of token ids
        B, T = idx.shape

        aux_loss_total = 0 # varible to carry accumulated aux_loss from ech transformer block

        # this checks if the input sequence is longer than the block size
        assert T <= self.config.seq_len, f"Cannot forward sequence of length {T}, sequence length is only {self.config.seq_len}"

        # this creates the embedding table for the token ids.
        token_embd = self.transformer.wte(idx) # (B, T, n_embd)
        
        # apply the transformer blocks. each block applies layer norm, self-attention, residual connection, layer norm, MoE layer, residual connection
        x = token_embd
        for block in self.transformer.h:
            x, aux_loss = block(x)
            aux_loss_total += aux_loss
        
        # apply layer norm to the output of the last transformer block
        x = self.transformer.ln_f(x)

        # apply the final linear layer to get the logits for the next token prediction
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # if targets are provided, calculate the loss
        loss = None
        if targets is not None:
            # Pytorch's cross-entropy loss expects the logits to be of shape (B*T, vocab_size) and the targets to be of shape (B*T). So we need to reshape the logits and targets to match this shape.
            # reshape the logits: (B, T, vocab_size) -> (B*T, vocab_size) to match the shape of the targets: (B, T) -> (B*T) and then calculate the cross-entropy loss
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        total_loss = main_loss + (.01*aux_loss_total)
        
        return logits, total_loss


#%%



