#%%
#utility file to create model
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
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
    
        # apply rotation and transpose. the reshaping to (B, n_heads, seq_len, dim_heads) is to accommodate the matrix multiplication of k, q, and v along the desiored dimensions
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
class TopKMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.k = config.k
        self.seq_len = config.seq_len
    
        # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each expert. the logits will have shape (batch_size, seq_len, num_experts) 
        self.gate_linear = nn.Linear(config.n_embd, config.num_experts, bias=False)

        # this is a learnable parameter that will be used to scale the noise added to the logits of each expert. The allows the noise added to each expert to be "customized" and dynamic. It adapts during training depending on the expert. It is initialized to zero and will be learned during training.
        self.noise_weight = nn.Parameter(torch.zeros(config.num_experts)) 

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
        zeros = torch.full_like(logits_noisy, float('-inf')) 

        # scatter fills the zeros tensor with the top_k_logits at the indices of the top_k_indices along the last dimension (-1). This creates a sparse matrix where only the top-k logits are kept and the rest are filled with negative infinity.
        sparse_logits_noisy = zeros.scatter(-1, top_k_indices_noisy, top_k_logits_noisy)
        gated_weights = F.softmax(sparse_logits_noisy, dim=-1)

        return gated_weights, top_k_indices_noisy, top_k_logits_noisy


# %%
# this class implements the full MoE layer. It uses the TopKMoEGate class to compute the gated weights and the top-k indices, and then applies each expert to the input based on these indices. The output is a weighted sum of the expert outputs, where the weights are the gated weights.
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.sq_len = config.seq_len
        self.k = config.k
    
        # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each expert. the logits will have shape (batch_size, seq_len, num_experts) 
        self.gate = TopKMoEGate(config)

        # Create a list of experts, each expert is an instance of the ExpertMoESwiglu class
        self.experts = nn.ModuleList([ExpertMoESwiglu(config) for _ in range(self.num_experts)])
        torch.manual_seed(42)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Get the gated weights and top-k indices from the gate
        gated_weights, top_k_indices, top_k_logits = self.gate(x)

        # Initialize the fianl output tensor
        final_output = torch.zeros_like(x)
        # print(f'\ninput x shape: {x.shape} \n{x}\n')

        # flatten the input to (batch_size * seq_len, n_embd) for batch processing
        x_flat = x.view(batch_size*seq_len, -1) 
        # print(f'\ninput x_flat shape: {x_flat.shape} \n{x_flat}\n')

        # flatten the gated weights to (batch_size * seq_len, num_experts) for batch processing
        gated_weights_flat = gated_weights.view(batch_size*seq_len, self.num_experts)  

        # Iterate over each expert and apply it to the input
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k. the mask will have shape (batch_size, seq_len) and will be True for the tokens where expert i is in the top_k indices.
            # print(f'\nExpert {i} with x_flat input shape {x_flat.shape}\n')
            # print(f'top_k_indices shape: {top_k_indices.shape} \n{top_k_indices}\n')
            expert_mask = (top_k_indices == i).any(dim=-1)
            # print(f'expert_mask shape: {expert_mask.shape} \n{expert_mask}\n')

            # flatten the mask to match the shape of the flattened input x_flat. Note that the shape of flat_mask is a one dimensional (batch_size*seq_len). x_flat has shape (batch_size * seq_len, n_embd). each row in x_flat is a token in the sequence. Pytorch applies the mask to the rows of x_flat based on shape matching.
            flat_mask = expert_mask.view(-1) # (batch_size * seq_len)
            # print(f'flat_mask shape: {flat_mask.shape} \n{flat_mask}\n')

            if flat_mask.any():
                # Apply the expert to the inputs selected by the mask. x_flat[flat_mask] picks the rows(tokens) of x_flat where the mask is True. This allows us to activate the expert only for the tokens where the expert is in the top_k indices.
                expert_input = x_flat[flat_mask] # (number of tokens where expert i is in top_k, n_embd)
                # print(f'\nexpert_input shape: {expert_input.shape} \n{expert_input}\n')
                
                # apply expert i to the expert_input. Again, note that based on the mask described above, epxert i is applied only to the tokens where it is in the top_k indices.
                expert_output = expert(expert_input) # (number of tokens where expert i is in top_k, n_embd)
                # print(f'expert_output shape: {expert_output.shape} \n{expert_output}\n')

                # Now we need to scale the expert output by the gated weights for the tokens where the expert is in the top_k indices. gated_weights_flat has shape (batch_size * seq_len, num_experts). We apply the same mask as we used to create expert_input to select all rows from gated_weights_flat where expert i is in the top_k indices, then we select the ith column. This returns the weighting for expert i that is to be applied to the tokens where expert i is in the top_k indices. This is the same as selecting all the non_zero values in the ith column of gated_weights_flat. We  then use unsqueeze(1) to add a dimension to create a column vector of shape (number of tokens where expert i is in top_k, 1). this allows  multiplication with the expert_output which has shape (number of tokens where expert i is in top_k, n_embd). 
                # print(f'gated_weights_flat shape: {gated_weights_flat.shape} \n{gated_weights_flat}\n')
                expert_weights = gated_weights_flat[flat_mask, i].unsqueeze(1)  # (number of tokens where expert i is in top_k, 1)
                # print(f'expert_weights shape: {expert_weights.shape} \n{expert_weights}\n')

                # Scale the expert_output by expert_weights.
                expert_output_weighted = expert_output * expert_weights # (number of tokens where expert i is in top_k, n_embd)
                # print(f'expert_output_weighted shape: {expert_output_weighted.shape} \n{expert_output_weighted}\n')

                # Now we need to add the expert_output_weighted to final_output at positions where the expert is in the top_k indices. We use expert_mask to select the rows where expert i is in the top_k indices. Note that here we use expert_mask (not flat_mask) with shape (batch_size, seq_len, hidden_dim) to match the shape of final_output. final_output will have shape (batch_size, seq_len, n_embd), the same as input x.
                # print(f'final_output shape before adding expert_output_weighted:{final_output.shape} \n{final_output}\n')

                # the huggingface implementation uses .squeeze(1) to remove any singleton dimensions from  the expert_output_weighted tensor. Not sure why this is needed. I tried removing it and the shapes were still compatible and the result the same
                
                # final_output[expert_mask] += expert_output_weighted.squeeze(1) # (batch_size, seq_len, n_embd)
                
                
                expert_contribution = torch.zeros_like(final_output)
                expert_contribution[expert_mask] = expert_output_weighted.squeeze(1)
                final_output = final_output + expert_contribution
                
                
                # print(f'final_output shape after adding expert_output_weighted:{final_output.shape} \n{final_output}\n')

                ## just a test to see if .squeeze(1) is necessary
                # final_test = torch.zeros_like(x)
                # final_test[expert_mask] += expert_output_weighted
                # print(f'{torch.allclose(final_output, final_test)}')

            # break

        return final_output, top_k_indices

#%%
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_out, top_k_indices = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, top_k_indices

# %%
class CreateMoE(nn.Module):
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

    def forward(self, idx, targets=None):
        # idx is the input sequence of token ids

        B, T = idx.shape

        # this checks if the input sequence is longer than the block size
        assert T <= self.config.seq_len, f"Cannot forward sequence of length {T}, sequence length is only {self.config.seq_len}"

        # this creates the embedding table for the token ids.
        token_embd = self.transformer.wte(idx) # (B, T, n_embd)
        
        # apply the transformer blocks. each block applies layer norm, self-attention, residual connection, layer norm, MoE layer, residual connection
        x = token_embd
        top_k_all = []
        for block in self.transformer.h:
            x, top_k_indices = block(x)
            top_k_all.append(torch.flatten(top_k_indices))
        
        # apply layer norm to the output of the last transformer block
        x = self.transformer.ln_f(x)

        # apply the final linear layer to get the logits for the next token prediction
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # if targets are provided, calculate the loss
        loss = None
        if targets is not None:
            # Pytorch's cross-entropy loss expects the logits to be of shape (B*T, vocab_size) and the targets to be of shape (B*T). So we need to reshape the logits and targets to match this shape.
            # reshape the logits: (B, T, vocab_size) -> (B*T, vocab_size) to match the shape of the targets: (B, T) -> (B*T) and then calculate the cross-entropy loss
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss, top_k_all


#%%
#--------------------------------------------------------------------------------

# # from claude

# class MoELayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_experts = config.num_experts
#         self.sq_len = config.seq_len
#         self.k = config.k
    
#         # Create a linear layer to project the multi-head attention output to the number of experts
#         self.gate = TopKMoEGate(config)
#         # Create a list of experts
#         self.experts = nn.ModuleList([ExpertMoESwiglu(config) for _ in range(self.num_experts)])
#         torch.manual_seed(42)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
        
#         # Get the gated weights and top-k indices from the gate
#         gated_weights, top_k_indices, top_k_logits = self.gate(x)

#         # Initialize the final output tensor - use explicit device and dtype
#         final_output = torch.zeros(batch_size, seq_len, x.size(-1), 
#                                  device=x.device, dtype=x.dtype)

#         # flatten the input - ensure contiguous memory layout
#         x_flat = x.contiguous().view(batch_size * seq_len, -1)
        
#         # flatten the gated weights
#         gated_weights_flat = gated_weights.contiguous().view(batch_size * seq_len, self.num_experts)

#         # Process all experts and accumulate results
#         for i, expert in enumerate(self.experts):
#             # Create mask for current expert
#             expert_mask = (top_k_indices == i).any(dim=-1)
#             flat_mask = expert_mask.contiguous().view(-1)

#             if flat_mask.any():
#                 # Get expert input
#                 expert_input = x_flat[flat_mask]
                
#                 # Apply expert
#                 expert_output = expert(expert_input)
                
#                 # Get weights for this expert
#                 expert_weights = gated_weights_flat[flat_mask, i].unsqueeze(1)
                
#                 # Scale expert output
#                 expert_output_weighted = expert_output * expert_weights
                
#                 # SAFE ACCUMULATION: Use scatter_add instead of indexing
#                 # First, get the indices where this expert should contribute
#                 mask_indices = expert_mask.nonzero(as_tuple=False)  # Shape: (num_active, 2)
                
#                 if mask_indices.size(0) > 0:
#                     # Create indices for scatter_add
#                     batch_indices = mask_indices[:, 0]  # batch dimension
#                     seq_indices = mask_indices[:, 1]    # sequence dimension
                    
#                     # Use advanced indexing to safely add expert contributions
#                     for j, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
#                         final_output[b_idx, s_idx] = final_output[b_idx, s_idx] + expert_output_weighted[j]

#         return final_output, top_k_indices


# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(config.n_embd)
#         self.attn = CausalSelfAttention(config)
#         self.ln_2 = nn.LayerNorm(config.n_embd)
#         self.moe = MoELayer(config)

#     def forward(self, x):
#         # Ensure input is contiguous
#         x = x.contiguous()
        
#         # Apply attention with residual connection
#         attn_out = self.attn(self.ln_1(x))
#         x = x + attn_out
        
#         # Apply MoE with residual connection  
#         moe_out, top_k_indices = self.moe(self.ln_2(x))
#         x = x + moe_out
        
#         return x.contiguous(), top_k_indices


# class CreateMoE(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.vocab_size, config.n_embd),
#             h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#             ln_f = nn.LayerNorm(config.n_embd)
#         ))
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#         # Weight tying - ensure this doesn't create problematic views
#         # Instead of direct assignment, copy the weights
#         with torch.no_grad():
#             self.lm_head.weight.copy_(self.transformer.wte.weight)
        
#         # Make them share the same parameter object (proper weight tying)
#         self.lm_head.weight = self.transformer.wte.weight

#         # initialization
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         std = 0.02
#         if isinstance(module, nn.Linear):
#             if hasattr(module, "_am_expert"):
#                 std = 0.01
#             if hasattr(module, 'NANOGPT_SCALE_INIT'):
#                 std *= (2 * self.config.n_layer) ** -0.5
#             nn.init.normal_(module.weight, mean=0.0, std=std)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(self, idx, targets=None):
#         B, T = idx.shape
#         assert T <= self.config.seq_len, f"Cannot forward sequence of length {T}, max is {self.config.seq_len}"

#         # Ensure idx is contiguous and not a view
#         idx = idx.contiguous()
        
#         # Create token embeddings
#         token_embd = self.transformer.wte(idx)  # (B, T, n_embd)
        
#         # Apply transformer blocks
#         x = token_embd.contiguous()  # Ensure contiguous
#         top_k_all = []
        
#         for block in self.transformer.h:
#             x, top_k_indices = block(x)
#             top_k_all.append(torch.flatten(top_k_indices))
        
#         # Final layer norm
#         x = self.transformer.ln_f(x)
        
#         # Get logits
#         logits = self.lm_head(x)  # (B, T, vocab_size)
        
#         # Calculate loss if targets provided
#         loss = None
#         if targets is not None:
#             # Ensure targets is contiguous
#             targets = targets.contiguous()
#             loss = F.cross_entropy(
#                 logits.contiguous().view(-1, logits.size(-1)), 
#                 targets.view(-1)
#             )
        
#         return logits, loss, top_k_all
