#%%
#utility file to create model
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from deepspeed.moe.layer import MoE
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
        self.ln_1 = nn.Linear(config.n_embd, self.hidden_dim) 
        self.ln_2 = nn.Linear(config.n_embd, self.hidden_dim)
        self.c_proj = nn.Linear(self.hidden_dim, config.n_embd)
        # torch.manual_seed(42)

    def forward(self, x):
        x =self.ln_1(x) * F.silu(self.ln_2(x))  # this is Swiglu activation
        x= self.c_proj(x)
        return x
    



# %%
# this class implements the full MoE layer in a form that is compatible with DeepSpeed. Note that DeepSpeed intenally takes care of the routing and expert selection, so we only need to define the experts and the gate projection. As compared to the the implementation in aae_model_rotary_moe.py, this class does not implement the routing and and top_k selection. Deepseed handles this internally.
class DeepSpeedMoeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        
        self.moe = MoE(
            hidden_size=config.n_embd,
            expert=lambda config: ExpertMoESwiglu(config),
            num_experts=config.num_experts,
            ep_size=8,
            k=config.k,
            capacity_factor=1.25,
            eval_capacity_factor=1.0,

            #use_residual=False because in a Transformer MoE layer, the residual connection is handled outside the MoE block, at the Transformer block level, not inside the MoE itself.
            use_residual=False, 
            
            noisy_gate_policy=None, # we will handle noise ourselves
            
            gate=self.gate_fn # gate function that DeepS will use to select experts. Defined below
        )


    def gate_fn(self, x):
        batch_size, seq_len, _ = x.shape

        # flatten the input to (batch_size*seq_len, n_embd)
        x_flat = x.view(batch_size*seq_len, self.n_embd)  
        logits = self.gate_proj(x_flat) # (batch_size*seq_len, num_experts)

        # add noise to the logits.
        if self.training:
            noise = torch.randn_like(logits) * self.noisy_std
            logits_noisy = logits + noise

        # apply per expert weights to logits
        logits_noisy_weighted = logits_noisy * self.expert_weights


#%%
@dataclass
class GPTConfig:
    seq_len: int = 1024 # max sequence length
    # setting vocab size to 50304 rather than 50257 (the size of the gpt2 vocab) because this is a much more efficient number (divisible by many powers of 2) for gpu kernels and computations. The extra tokens are just padding tokens that are not used in the model. The model will learn to ignore them. this is a tradeoff between memory and performance. 
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_experts = 4
    k = 2
    noisy_std = 1.0
config = GPTConfig()
test = DeepSpeedMoeLayer(config)
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
        x = x + self.moe(self.ln_2(x))
        return x

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

    # this Karpathy's weight initialization code that I dont really follow
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is the input sequence of token ids

        B, T = idx.shape

        # this checks if the input sequence is longer than the block size
        assert T <= self.config.seq_len, f"Cannot forward sequence of length {T}, sequence length is only {self.config.seq_len}"

        # this creates the embedding table for the token ids.
        token_embd = self.transformer.wte(idx) # (B, T, n_embd)
        
        # apply the transformer blocks. each block applies layer norm, self-attention, residual connection, layer norm, MoE layer, residual connection
        x = token_embd
        for block in self.transformer.h:
            x = block(x)
        
        # apply layer norm to the output of the last transformer block
        x = self.transformer.ln_f(x)

        # apply the final linear layer to get the logits for the next token prediction
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # if targets are provided, calculate the loss
        loss = None
        if targets is not None:
            # Pytorch's cross-entropy loss expects the logits to be of shape (B*T, vocab_size) and the targets to be of shape (B*T). So we need to reshape the logits and targets to match this shape.
            # reshape the logits: (B, T, vocab_size) -> (B*T, vocab_size) to match the shape of the targets: (B, T) -> (B*T) and then calculate the cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


#%%



