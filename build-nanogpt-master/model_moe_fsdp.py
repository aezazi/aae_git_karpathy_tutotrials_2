#%%
# imports
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
        self.batch_size = config.batch_size
        self.load_balance_scale = config.load_balance_scale
    
        # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each expert. the logits will have shape (batch_size, seq_len, num_experts) 
        self.gate_linear = nn.Linear(config.n_embd, config.num_experts, bias=False)

        # this is a learnable parameter that will be used to scale the noise added to the logits of each expert. This allows the noise added to each expert to be "customized" and dynamic. It adapts during training depending on the expert. It is initialized to zero and will be learned during training.
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
        # x_flat has shape (batch_size*sequence_length, embedding dimension) and is the output of the multi-head attention layer after being flattened in foreard method of the MoELayer
        token_count, n_embd = x_flat.shape

        # just a check to make sure the shape manipulations are consistent with original input
        assert n_embd == self.n_embd, f"Expected embedding dim {self.n_embd}, got {n_embd}"
        # assert token_count == self.batch_size*self.seq_len, f"token_count {self.batch_size*self.seq_len}, got {token_count}"

        # project the multi-head attention output n_embd to the number of experts
        logits = self.gate_linear(x_flat) # (batch_size*seq_len, num_experts)

        # The global noise to be added to the logits of each expert. This is a random noise tensor of the same shape as the logits multiplied by noisy_std to create desired level of variance
        noise = torch.randn_like(logits) * self.noisy_std # (batch_size*seq_len, num_experts)

        # Per-expert noise scaling using self.noise_weights. 
        noise = noise * self.noise_weight

        # Add the noise to the logits. 
        logits_noisy = logits + noise  # (batch_size*seq_len, num_experts)

         # Get the top-k logits and their corresponding indices. The pytorch top_k method returns the top-k values and their indices along the last dimension (num_experts). In each batch, for each token in the sequence, it selects the top-k logits and their indices from the logits produced by each expert. Note that the top_k ids are global and not necessarily on this gpu.
        top_k_logits_noisy, top_k_ids_noisy = logits_noisy.topk(self.k, dim=-1)  # (batch_size* seq_len, top_k) 

        # We want sparse matrices. We achieve this by keeping the top-k logits for each token and filling the rest with a value that represents "no contribution" (like negative infinity).  This is done to ensure that the softmax function will ignore these values when computing the weights for each expert. So only the values produced by the top-k experts will contribute to the final output. We implement this by first creating a tensor of the same shape as the logits filled with negative infinity, and then using the scatter function to fill in the top-k logits at the indices of the top-k experts. Note that in this context, softmax is being used to compute "weights" for each expert not probabilities as for multiclass classification. Its a subttle difference but I think important to note.
        
        #full_like clones a tensor and fills it with a specified value (like infinity).
        zeros = torch.full_like(logits_noisy, float('-inf')) 

        # scatter fills the zeros tensor with the top_k_logits at the indices of the top_k_ids_global_noisy along the last dimension (-1). This creates a sparse matrix where only the top-k logits are kept and the rest are filled with negative infinity.
        sparse_logits_noisy = zeros.scatter(-1, top_k_ids_noisy, top_k_logits_noisy)

        # Note top_k_gated_weights has shape #(B*seq_len, num_experts). The top_k experts will have weights that sum to 1, the other experts will have weights-inf
        top_k_gated_weights = F.softmax(sparse_logits_noisy, dim=-1) 

        # compute load balance loss using pre-noise logits
        load_balance_loss = self._compute_load_balance_loss(logits)

        return top_k_gated_weights, top_k_ids_noisy, load_balance_loss


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
        """
        Create MoE layer
        """
        batch_size, seq_len, _ = x.shape

        # flatten the input to (batch_size * seq_len, n_embd) for batch processing
        x_flat = x.view(batch_size*seq_len, -1) 
        # print(f'\ninput x_flat shape: {x_flat.shape} \n{x_flat}\n')

        # Get the top_k_gated weights and top-k indices from the gate. 
        top_k_gated_weights_flat, top_k_indices, load_balance_loss = self.gate(x_flat)
        # print(f'[DEBUG] top_k_gated_weights_flat shape: {top_k_gated_weights_flat.shape}')
        # print(f'[DEBUG] top_k_indices shape: {top_k_indices.shape}')

        # Initialize the final output tensor
        final_output_flat = torch.zeros_like(x_flat)
        # print(f'\ninput x shape: {x.shape} \n{x}\n')

        # Iterate over each expert and apply it to the input
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k. the mask will have shape (batch_size*seq_len) and will be True for the tokens where expert i is in the top_k indices.
            # print(f'\n[DEGUG] Expert {i} with x_flat input shape {x_flat.shape}\n')
            # print(f'[DEGUG] top_k_indices shape: {top_k_indices.shape} \n{top_k_indices}\n')
            expert_mask_flat = (top_k_indices == i).any(dim=-1)
            # print(f'[DEGUG] expert_mask_flat shape: {expert_mask_flat.shape} \n{expert_mask_flat}\n')

            
            if expert_mask_flat.any():
                # Apply the expert to the inputs selected by the mask. x_flat[expert_mask_flat] picks the rows(tokens) of x_flat where the mask is True. This allows us to activate the expert only for the tokens where the expert is in the top_k indices.
                expert_input = x_flat[expert_mask_flat] # (number of tokens where expert i is in top_k, n_embd)
                # print(f'\nexpert_input shape: {expert_input.shape} \n{expert_input}\n')
                
                # apply expert i to the expert_input. Again, note that based on the mask described above, epxert i is applied only to the tokens where it is in the top_k indices.
                expert_output = expert(expert_input) # (number of tokens where expert i is in top_k, n_embd)
                # print(f'expert_output shape: {expert_output.shape} \n{expert_output}\n')

                # Now we need to scale the expert output by the gated weights for the tokens where the expert is in the top_k indices. gated_weights_flat has shape (batch_size * seq_len, num_experts). We apply the same mask as we used to create expert_input to select all rows from gated_weights_flat where expert i is in the top_k indices, then we select the ith column. This returns the weighting for expert i that is to be applied to the tokens where expert i is in the top_k indices. This is the same as selecting all the non_zero values in the ith column of gated_weights_flat. We  then use unsqueeze(1) to add a dimension to create a column vector of shape (number of tokens where expert i is in top_k, 1). this allows  multiplication with the expert_output which has shape (number of tokens where expert i is in top_k, n_embd). 
                # print(f'gated_weights_flat shape: {gated_weights_flat.shape} \n{gated_weights_flat}\n')
                expert_weights = top_k_gated_weights_flat[expert_mask_flat, i].unsqueeze(1)  # (number of tokens where expert i is in top_k, 1)
                # print(f'expert_weights shape: {expert_weights.shape} \n{expert_weights}\n')

                # Scale the expert_output by expert_weights.
                expert_output_weighted = expert_output * expert_weights # (number of tokens where expert i is in top_k, n_embd)
                # print(f'[DEBUG] expert_output_weighted shape: {expert_output_weighted.shape}')

                # Now we need to add the expert_output_weighted to final_output_flat at positions where the expert is in the top_k indices. 
                
                # the huggingface implementation uses .squeeze(1) to remove any singleton dimensions from  the expert_output_weighted tensor. Not sure why this is needed. I tried removing it and the shapes were still compatible and the result the same

                # print(f'[DEBUG] final_output_flat[expert_mask_flat] shape: {final_output_flat[expert_mask_flat].shape}')
                final_output_flat[expert_mask_flat] += expert_output_weighted  # (batch_size*seq_len, n_embd)
                
                # print(f'[DEBUG] final_output_flat shape after adding expert_output_weighted:{final_output_flat.shape}')
                # print(f'[DEBUG] final_output_flat sample:\n{final_output_flat[0:5,0:7]}')

                
                ## just a test to see if .squeeze(1) is necessary
                # final_test = torch.zeros_like(x)
                # final_test[expert_mask] += expert_output_weighted
                # print(f'{torch.allclose(final_output, final_test)}')

        # reshape final output back to shape (batch_size, seq_len, n_embd)    
        final_output = final_output_flat.view(batch_size, seq_len, -1)
        return final_output, top_k_indices, load_balance_loss

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
        moe_out, top_k_indices, load_balance_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, top_k_indices, load_balance_loss

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
        load_balance_losses = []
        for block in self.transformer.h:
            x, top_k_indices, load_balance_loss = block(x)
            top_k_all.append(torch.flatten(top_k_indices))
            load_balance_losses.append(load_balance_loss)
        
        # apply layer norm to the output of the last transformer block
        x = self.transformer.ln_f(x)

        # apply the final linear layer to get the logits for the next token prediction
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # if targets are provided, calculate the loss
        total_load_balance_loss = sum(load_balance_losses) / len(load_balance_losses)
        loss = None
        if targets is not None:
            # Pytorch's cross-entropy loss expects the logits to be of shape (B*T, vocab_size) and the targets to be of shape (B*T). So we need to reshape the logits and targets to match this shape.
            # reshape the logits: (B, T, vocab_size) -> (B*T, vocab_size) to match the shape of the targets: (B, T) -> (B*T) and then calculate the cross-entropy loss
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
            total_loss = loss + total_load_balance_loss
        
            return logits, total_loss, top_k_all
        
        return logits, loss, top_k_all


#%%
