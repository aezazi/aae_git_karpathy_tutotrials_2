#%%
import os
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
import time


#%%
# Set the device      
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# note that the variable "device" is a pytorch device object. It is not a string. So you cannot use it as a string in an if statements. You have to use the .type attribute of the device object to get the type of the device as a string.
print(f'type of device object: {type(device)}')
print(f'device type as string: {device.type}')

# %%
# This is the configuration for the GPT model. It defines the hyperparameters for the model. The block size is the maximum sequence length, vocab size is the size of the vocabulary, n_layer is the number of transformer blocks, n_head is the number of attention heads, and n_embd is the embedding dimension. 
"""
Note that in the initialization of the network in the MLP class, we are multiplying n_embd (the dimensions of the original embeddings) by 4. So for the inner layers, the dimensionality of the model is 384 * 4 =1536. 
"""

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    # setting vocab size to 50304 rather than 50257 (the size of the gpt2 vocab) because this is a much more efficient number (divisible by many powers of 2) for gpu kernels and computations. The extra tokens are just padding tokens that are not used in the model. The model will learn to ignore them. this is a tradeoff between memory and performance. 
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# instantiate and check the config
config = GPTConfig()
print(f'GPTConfig instantiated with block size: {config.block_size}, vocab size: {config.vocab_size}, n_layer: {config.n_layer}, n_head: {config.n_head}, n_embd: {config.n_embd}')

#%%
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 

        # key, query, value, projections for all heads, but in a batch. The output of the linear layer is 3 times the size of the input. I'm not what the multiplication by 3 is for. presumably because we later divide the output of the linear layer into 3 parts for q, k, v

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # input is a batch of sequences of embeddings
        B, T, C = x.size()

        # split the embeddings into key, query, value
        # the first 2 dimensions are the batch and sequence length. the last dimension is the embedding dimension
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs  e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the transformer
        qkv = self.c_attn(x) # (B, T, 3 * C)
        # print(qkv.shape)

        # divide the output of the linear layer into 3 parts for q, k, v
        q, k, v = qkv.chunk(3, dim=-1)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)

        # q, k, v = qkv.split(self.n_embd, dim=2)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)

        # Karpathy explains the purpose of the following to be to make the process more efficient in Pytorch by splitting the channels into multiple heads. Each head is a slice of the channels. This allows for more parallelization and less memory usage.
        # reshape q, k, v for multi-head attention and transpose for dot product: (B, nh, T, hs)
        # B is the batch size, T is the sequence length, nh is the number of heads, hs is the head size (C // n_head)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # print('-'*50)

        # Pytorch implementation of Flash attention algorithim. This is the scaled dot-product attention built-in pytorch function. It takes the dot product of the query and key, scales it by the square root of the head size, and then applies a softmax to get the attention weights. The attention weights are then multiplied by the value to get the output. the is_causal=True argument ensures that the attention is only applied to the left of the current position in the sequence (i.e. it is causal). This is done by applying a mask to the attention weights. See Karpathy's video tutorial at 2:00:00 for more details. 
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        
        # transpose back to (B, T, nh*hs) and combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

#%%
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
         # multiply by 4 for additional dimensions and computational power
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear( 4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

#%%
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# %%
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), #positional embeddings
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
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # this creates the position ids for the input sequence. It's just a range of integers from 0 to T (the sequence length)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 

        # This creates an embedding table for the postions. It's just a lookup table for the position embeddings.
        pos_embd = self.transformer.wpe(pos) # (T, C)

        # this creates the embedding table for the token ids.
        token_embd = self.transformer.wte(idx) # (B, T, n_embd)

        # Position embeddings are added to the token embeddings to give the model information about the order of the tokens in the sequence. Note that the position embeddings are the same for all sequences of the same length, but the token embeddings are different for each sequence. Also position embeddings have shape (T, C) and token embeddings have shape (B, T, C). Pytorch broadcasts the position embeddings to the batch size.
        x = token_embd + pos_embd # (B, T, n_embd)

        # apply the transformer blocks. each block applies layer norm, self-attention, residual connection, layer norm, MLP, residual connection
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


    # code below is from Karparpathy without any additional comments from me. It loads the weights from  the huggingface model into the GPT model. 
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
   


# %%
#Instantiate the model and implement torch.compile if cuda is available.
model = GPT(GPTConfig())

# Implement torch.compile.
# if cuda is available, use torch.compile to optimize the model for training on GPUs. This is a performance optimization that allows for more efficient training on GPUs. It uses the PyTorch JIT compiler to optimize the model for the specific hardware and software configuration. This is done to improve performance and reduce memory usage. we use bfloat16 precision for the forward pass and use torch.compile. See Karpathy's tutorial at 1:24:00 and 1:49:00 for details

if device.type == 'cuda':
    print(f'using device: {device}')
    torch.set_float32_matmul_precision('high')
    print('using high precision for cuda')
    model = torch.compile(model)
    print('torch.compile applied to model')

model.to(device)
# Check model is on which device. 
print(f'model and parameters are on device: {next(model.parameters()).device}')


#%%
# NOTE: I moved the code for optimizer configuration to a separate file called aae_utils.py.
from aae_utils import ConfigureOptimizer

# Instantiate and create the optimizer.
base_lr = 6e-4
optimizer = ConfigureOptimizer(model).create_optimizer(weight_decay=0.01, learning_rate=base_lr, device_type=device.type)

print('Optimizer initialized successfully!')

# %%
# Instantiate the dataloader and load the data.
# NOTE: I moved the code for the dataloader to a separate file called aae_utils.py. 
from aae_utils import DataLoaderLite

# initialize the dataloader based on the device type. The batch size and sequence length are set based on the device type and my experiments.

# NOTE: using device.type to get device as string for if statements.
if device.type == 'cuda':
    train_loader = DataLoaderLite(B=16, T=1024)
elif device.type == 'mps':
    train_loader = DataLoaderLite(B=8, T=1024)
else:
    train_loader = DataLoaderLite(B=8, T=512)

# we want to match the batch size of 0.5M used in the GPT2. Our GPUs can't handle that. So we will use a smaller batch size and accumulate gradients over multiple steps to get the same effect. See the training loop below for details on implementing gradient accumulation.
effective_batch_size_desired =524288 # 2^19 ~ .5M to match the original GPT-2 paper. 

assert effective_batch_size_desired % (train_loader.B * train_loader.T) == 0, f"effective batch size {effective_batch_size_desired} is not divisible by batch size {train_loader.B} and sequence length {train_loader.T}"

# this is the desired number of micro steps to accumulate gradients over. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU.
accumulation_steps = effective_batch_size_desired // (train_loader.B * train_loader.T) 
print(f"accumulation steps desired: {accumulation_steps}")


#%%
# create scheduler and launch training loop.
# NOTE: I moved the code for the scheduler to a separate aae_utils.py file.
from aae_utils import CosineLearingRateScheduler

training_steps = 50

# define the scheduler parameters
T_max = training_steps # the number of iterations over which lr is reduced to the minimum
max_lr = base_lr
min_lr = max_lr * 0.1
warm_up_steps = .1 * T_max 
restart = False # whether to use cosine annealing with restarts or not
T_0 = T_max // 3 # if using restarts, the number of iterations over which lr is reduced to the minimum before restart
T_mult = 2 # the factor by which T_0 is multiplied at each restart.

# instantiate and create learning rate scheduler
scheduler = CosineLearingRateScheduler(optimizer=optimizer, T_max=T_max, restart=restart, warm_up_steps=warm_up_steps, max_lr=max_lr, min_lr=min_lr, T_mult=T_mult, T_0=T_0)

print('Scheduler initialized successfully!')


#%%
# Run the training loop.
# if cuda or mps is not available, use just 2 accumulation steps. 
if device.type != 'cuda' or device.type == 'mps': 
    accumulation_steps = 2

loss_list= []
for step in range(training_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum  = 0.0
    for micro_step in range(16):
        # this is a gradient accumulation step. We accumulate gradients over desired accumalation steps before updating the weights. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU. 
        x, y = train_loader.next_batch()
        
        # if the decive is 'cuda', we use autocast to use bfloat16 precision for the forward pass. This is a performance optimization for training on GPUs.
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x.to(device), y.to(device))
        else:
            logits, loss = model(x.to(device), y.to(device))
        
        loss = loss / accumulation_steps # divide the loss by the number of micro steps to get the average loss of the accumulated micro steps
        loss_accum += loss.detach() # accumulate the loss over the micro steps. Look at Pytorch documentation for more details on tensor.detach() vs. tensor.item()
        loss.backward()

    # clip the gradients to prevent exploding gradients
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.set_lr(step)

    # synchronize the device to make sure all operations are complete before measuring time
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = train_loader.B * train_loader.T * accumulation_steps/ (t1 - t0)
    print(f"Step {step}, Loss: {loss_accum.item()}, LR: {optimizer.param_groups[0]['lr']}, norm: {norm:.4f}, Time: {dt:.2f}ms, Tokens/s: {tokens_per_sec:.2f}")
    
    if step % 5 == 0:
        loss_list.append(loss_accum.item())
    

#%%
# Plotting the learning rate schedule

import matplotlib.pyplot as plt
plt.figure(figsize=(100, 40))
plt.plot(range(training_steps), scheduler.lrs, marker='o')
plt.title('Learning Rate Schedule: Warmup + CosineAnnealingWarmRestarts')
plt.xlabel('step')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()

plt.figure(figsize=(100, 40))
plt.plot(range(training_steps//5), loss_list, marker='o')
plt.title('Loss vs Step')
plt.xlabel('step')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

#%%
print(loss_list)
# %%
###################################################################################################

# code below were building blocks to arrive at code above here. Also code to generate from huggingface pretrained weights

###################################################################################################


# %%
# Set the number of sequences to generate and the maximum length of each sequence
num_return_sequences = 5
max_length = 30
model.eval()
model.to(device)

# %%
# encode the input text and convert it to a tensor
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
print(tokens)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)


# %%
# generate the sequences
torch.manual_seed(42)
torch.mps.manual_seed(42)

# while the size of the generated sequences is less than the maximum length, we will keep generating tokens
while x.size(1) < max_length:
    with torch.no_grad():
        outputs = model(x)
        logits = outputs[0][:, -1, :]  # get the logits for the last token
        probs = F.softmax(logits, dim=-1)  # apply softmax to get probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  
        ix = torch.multinomial(topk_probs, num_samples=1)  # sample from the top-k probabilities
        xcol = torch.gather(topk_indices, -1, ix)  # get the sampled token
        x = torch.cat((x, xcol), dim=1)  # append the sampled token to the input

# %%
# decode and print the generated sequences
# Note that with the pretrained model, the output is ok. Without the pretrained model, the output is just random gibberish before the model is trained.

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)

#%%
#---------- the code above was for generating text with a pretrained model -----------------
#---------------------code below is for training the model from scratch.--------------------

# %%
# load the text file for training
with open('input.txt', 'r') as f:
    text = f.read()

text = text[:1000]  # truncate to 10k characters for faster training

# encode the text and convert it to a tensor
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

# define batch size and sequence length
B, T = 4, 32  # batch size and sequence length

# buffer for batching. the +1 is to have a target token for the last token in the input sequence
buf = torch.tensor(tokens[:B*T + 1], dtype=torch.long)  
x = buf[:-1].view(B, T)  # input sequence
y = buf[1:].view(B, T)   # target sequence

# print(x)
# print(y)

# %%
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x.to(device), y.to(device))

# %%
# the loss here is  the loss before any training. A good test to see if the model is initialized correctly is to see if the loss is a number that makes sense. Before training, the probability of the model predicting any of the possible 50,257 tokens in the vocabulary is uniform. So the loss should be around -log(1/50257) = 10.9. (the negative log likelihood of any given token). If the loss is significantly different from this, then there is something wrong with the model initialization.
print(loss)
print(logits.shape)

# %%
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x.to(device), y.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x.to(device), y.to(device))
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}, Loss: {loss.item()}")


