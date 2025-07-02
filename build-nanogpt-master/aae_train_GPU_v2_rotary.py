#%%
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
import time
import numpy as np
import math
import csv

#%%
assert torch.cuda.is_available()  ,"This script is designed to run on CUDA devices only. Please ensure you have a compatible GPU."

# %%
# This is the configuration for the GPT model. It defines the hyperparameters for the model. The block size is the maximum sequence length, vocab size is the size of the vocabulary, n_layer is the number of transformer blocks, n_head is the number of attention heads, and n_embd is the embedding dimension. 
"""
Note that in the initialization of the network in the MLP class, we are multiplying n_embd (the dimensions of the original embeddings) by 4. So for the inner layers, the dimensionality of the model is 384 * 4 =1536. 
"""

@dataclass
class GPTConfig:
    seq_len: int = 1024 # max sequence length
    # setting vocab size to 50304 rather than 50257 (the size of the gpt2 vocab) because this is a much more efficient number (divisible by many powers of 2) for gpu kernels and computations. The extra tokens are just padding tokens that are not used in the model. The model will learn to ignore them. this is a tradeoff between memory and performance. 
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# instantiate and check the config
config = GPTConfig()

print(f'\nGPTConfig instantiated with block size: {config.seq_len}, vocab size: {config.vocab_size}, n_layer: {config.n_layer}, n_head: {config.n_head}, n_embd: {config.n_embd}')

class RotaryPosEmbed(nn.Module):
    def __init__(self, head_dim=768, seq_len=config.seq_len):
        super().__init__()
        self.head_dim = head_dim
        # self.get_theta()
        theta =  10_000 ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        self.register_buffer("theta", theta)
        position = torch.arange(0, seq_len, 1.0)
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

        # key, query, value, projections for all heads, but in a batch. The output of the linear layer is 3 times the size of the input. I'm not sure what the multiplication by 3 is for. presumably because we later divide the output of the linear layer into 3 parts for q, k, v

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.rot_embed = RotaryPosEmbed(config.n_embd/config.n_head) # rotary embedding
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
        q, k, v = qkv.chunk(3, dim=-1) # each has shape (B, T, C)


        # Karpathy explains the purpose of the following to be to make the training process more efficient in Pytorch by splitting the channels into multiple heads. Each head is a slice of the channels. This allows for more parallelization and less memory usage.
    
        # for rotary embedding, do not tranpose k and q to (B, nh, T, hs) until the rotation is applied
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
    
        # apply rotation and transpose
        k_rot = self.rot_embed.apply_rotation(x=k).transpose(1, 2)
        q_rot = self.rot_embed.apply_rotation(x=q).transpose(1, 2)
        
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        # Pytorch implementation of Flash attention algorithim. This is the scaled dot-product attention built-in pytorch function. It takes the dot product of the query and key, scales it by the square root of the head size, and then applies a softmax to get the attention weights. The attention weights are then multiplied by the value to get the output. the is_causal=True argument ensures that the attention is only applied to the left of the current position in the sequence (i.e. it is causal). This is done by applying a mask to the attention weights. See Karpathy's video tutorial at 2:00:00 for more details. 
        
        y = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True) # (B, nh, T, hs)
        
        # transpose back to (B, T, nh*hs) and combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear1(x) * F.silu(self.linear2(x))

#%%
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
         # multiply by 4 for additional dimensions and computational power
        # self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # self.gelu = nn.GELU(approximate='tanh')
        self.swiglu = SwiGLU(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear( 4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x= self.c_proj(self.swiglu(x))
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
        assert T <= self.config.seq_len, f"Cannot forward sequence of length {T}, block size is only {self.config.seq_len}"

        # this creates the embedding table for the token ids.
        token_embd = self.transformer.wte(idx) # (B, T, n_embd)
        
        # apply the transformer blocks. each block applies layer norm, self-attention, residual connection, layer norm, MLP, residual connection
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
# DDP setup
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Check if we are running in DDP mode. If so, we will initialize the process group and set the device for each process.

# a simple way to check whether your script is being run under Distributed Data Parallel (DDP) — specifically when using torchrun with a cuda GPU. Note that you can be in DDP mode even with a single GPU when using torchrun. 
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    print(f'\nRunning in Distributed Data Parallel (DDP) mode')
    # Note that LOCAL_RANK is the rank of the process on one given machine (when using multiple machine), while RANK is the rank of the process across all machines (when using multiple gpus on multiple machines). When using a setup with just one machine, LOCAL_RANK and RANK are the same. 
    init_process_group(backend='nccl') # initialize the process group for DDP
    ddp_rank = int(os.environ['RANK']) # get the rank of the current process
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # get the local rank of the current process
    ddp_world_size = int(os.environ['WORLD_SIZE']) # get the total number of processes
    
    # set the device to the local rank of the current process
    device = f'cuda:{ddp_local_rank}' 
    torch.cuda.set_device(device) # set the device for the current process

    # the master process will perform logging and saving checkpoints.
    master_process = (ddp_rank == 0)

# if not using DDP, just use the next best availabel option
else: 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"\nusing device: {device}")

torch.manual_seed(42) # set the random seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(42) # set the random seed for cuda for reproducibility


# %%
#Instantiate the model and implement torch.compile if cuda is available.

# if cuda is available, use torch.compile to optimize the model for training on GPUs. This is a performance optimization that allows for more efficient training on GPUs. It uses the PyTorch JIT compiler to optimize the model for the specific hardware and software configuration. This is done to improve performance and reduce memory usage. we use bfloat16 precision for the forward pass and use torch.compile. See Karpathy's tutorial at 1:24:00 and 1:49:00 for details

model = GPT(GPTConfig())

torch.set_float32_matmul_precision('high')
model.to(device)
use_compile = True # set to True to use torch.compile
model = torch.compile(model) if use_compile else model 

# wrap the model in DDP if using DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
    print(f'\nModel wrapped in DDP on device: {device}')

# get the raw model from the DDP wrapper. This is useful for accessing the model's parameters and methods directly. the raw_model is the actual model that we want to optimize. The DDP is just a wrapper that allows us to use distributed data parallelism.
raw_model = model.module if ddp else model 


#%%
# Instantiate the optimizer.
# NOTE: I moved the code for optimizer configuration to a separate file called aae_utils.py.
from aae_utils import ConfigureOptimizer

base_lr = 6e-4

# Note that we are using the raw model here, not the DDP wrapped model. This is because the DDP wrapper does not have the optimizer parameters. The raw model is the actual model that we want to optimize.
optimizer = ConfigureOptimizer(raw_model).create_optimizer(weight_decay=0.1, learning_rate = base_lr, device_type=device)

print(f'\nOptimizer initialized on GPU rank {ddp_rank}, device {device}')


# %%
# Instantiate the dataloader and load the data.
# NOTE: I moved the code for the dataloader to a separate file  aae_dataloader_til.py. 
from aae_dataloader_utils import DataLoaderShardMultiGPU

# initialize the dataloader for training and validation data. Batch size has to be be customized to fit the gpu being used.
B = 64 # batch size
T = 1024 # sequence length

train_loader = DataLoaderShardMultiGPU(B=B, T=T, process_rank = ddp_rank, num_processes=ddp_world_size, split='train')

val_loader = DataLoaderShardMultiGPU(B=B, T=T, process_rank = ddp_rank, num_processes=ddp_world_size, split='val')

# we want to match the batch size of 0.5M used in the GPT2. Our GPUs can't handle that. So we will use a smaller batch size and accumulate gradients over multiple steps to get the same effect. See the training loop below for details on implementing gradient accumulation.
effective_batch_size_desired =524288 # 2^19 ~ .5M to match the original GPT-2 paper. 


assert effective_batch_size_desired % (train_loader.B * train_loader.T * ddp_world_size) == 0, f"effective batch size {effective_batch_size_desired} is not divisible by batch size {train_loader.B} and sequence length {train_loader.T}"

# this is the desired number of micro steps to accumulate gradients over. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU.
accumulation_steps_desired = effective_batch_size_desired // (train_loader.B * train_loader.T * ddp_world_size) 

if master_process:
    print(f"\neffective batch size desired: {effective_batch_size_desired}")
    print(f"accumulation steps desired: {accumulation_steps_desired}")

#%%
# Instantiate the learning rate scheduler 
# NOTE: I moved the code for the scheduler to a separate aae_utils.py file.
from aae_utils import CosineLearingRateScheduler

# 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
training_steps = 19703

# define the scheduler parameters
# the number of iterations over which lr is reduced to the minimum
T_max = training_steps 

max_lr = base_lr # max learning rate
min_lr = max_lr * 0.1 # min learning rate

# modified from gpt paper per AK suggestion to be more aggresive with startup steps than paper
warm_up_steps = 300 

# whether to use cosine annealing with restarts or not
restart = False 

# if using restarts, the number of iterations over which lr is reduced to the minimum before restart
T_0 = T_max // 4 

T_mult = 3 # the factor by which T_0 is multiplied at each restart.

# instantiate and create learning rate scheduler
scheduler = CosineLearingRateScheduler(optimizer=optimizer, T_max=T_max, restart=restart, warm_up_steps=warm_up_steps, max_lr=max_lr, min_lr=min_lr, T_mult=T_mult, T_0=T_0)
print(f'\nScheduler initialized on GPU rank {ddp_rank}, of {ddp_world_size}\n')

#%%
# create log files, loggers, and evaluators to store training loss, learning rate, validation loss, hellaswag eval results, and generate sample text.
import aae_eval_log_utils as eval_log_utils
log_params = eval_log_utils.LogParamsFilesConfig(
    ddp = ddp,
    ddp_world_size = ddp_world_size,
    ddp_rank = ddp_rank,
    # ddp_local_rank = ddp_local_rank
    model = model,
    device = device,
    encoder = tiktoken.get_encoding('gpt2'),
    # optimizer = optimizer,
    val_loader = val_loader,
    loss_dir = "train_loss",
    hella_accu_dir = "hella_accuracy",
    learn_rate_dir = 'learn_rate_sched',
    train_loss_file = "train_loss.csv",
    hella_accu_file = "hellaswag_eval.csv",
    lr_file = "learning_rate.csv",
    step = 0,
    shard_idx = 0,
    loss_accum = 0.0,
    lr = 0.0
)

#%%
# Run the training loop.
model.train() # set the model to training mode

for step in range(training_steps):
    t0 = time.time()
    last_step = (step == training_steps - 1)

    # Main training loop
    model.train()
    optimizer.zero_grad()
    loss_accum  = 0.0
    micro_steps = accumulation_steps_desired # set the number of mirco steps to accumulate gradients over
    for micro_step in range(micro_steps):
        # this is a gradient accumulation step. We accumulate gradients over desired accumalation steps before updating the weights. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU. 
        x, y, shard_idx, tokens_abandoned = train_loader.next_batch()
        x, y = x.to(device), y.to(device) # move the data to the device. 

        # By default, ddp synchronizes the loss from each process after each micro step by taking an average of all the processes and making that average the loss for all the processes for that step. Its very inefficient to do this at each micro_step. So we want to only synchronize gradients among all the processes on the last micro step. See Karpathy's video tutorial at 2:57:00 for more details. The code below sets the require_backward_grad_sync attribute of the model to True only on the last micro step. 
        if ddp:
            model.require_backward_grad_sync = (micro_step == micro_steps - 1) 

        # we use autocast to use bfloat16 precision for the forward pass. This is a performance optimization for training on GPUs. The device must be cuda.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        
        # divide the loss by the number of micro steps to get the average loss of the accumulated micro steps
        loss = loss / micro_steps 
        
        # Look at Pytorch documentation for more details on tensor.detach() vs. tensor.item()
        loss_accum += loss.detach() 
        loss.backward()


    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip the gradients to prevent exploding gradients
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.set_lr(step)

    # synchronize the device to make sure all operations are complete before measuring time
    torch.cuda.synchronize()
    
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * micro_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    
    # update log_params, log traing loss and learning rate to file, print processing stats.
    if master_process:
        # update log_params
        log_params.step = step
        log_params.shard_idx = shard_idx
        log_params.loss_accum = round(loss_accum.item(), 7)
        log_params.lr = round(optimizer.param_groups[0]['lr'], 7)

        #log training loss and learning rate to file
        eval_log_utils.TrainLoss(log_params=log_params).log_training_loss()
        eval_log_utils.LearningRate(log_params=log_params).log_learning_rate()

        # print processing stats
        print(f"Step {step},  shard_idx: {shard_idx},  Loss: {loss_accum.item():.5f},  LR: {optimizer.param_groups[0]['lr']:.7f},  norm: {norm:.4f}, Time: {dt:.2f}sec,  Tokens/sec: {tokens_per_sec:,.0f}")

    # every x steps evaluate, print, and log hellaswag.
    if ((step > 0 and step % 250 == 0) or last_step):
        eval_log_utils.HellaSwag(log_params=log_params).log_print_hella_accuracy()

    # Every x steps, put the model in validation mode and use the validation dataset to compute loss. This is to help us catch any over fitting issues. 
    if step % 250 == 0 and step > 0:
        eval_log_utils.Validation(log_params=log_params).check_validation_loss()
    
    # every x steps generate from the model.
    if ((step % 500 == 0 and step > 0) or last_step):
        eval_log_utils.GenerateSample(log_params=log_params).generate(context="Hello, I'm a language model,", sample_max_length=32)

if ddp:
    destroy_process_group()

import sys; sys.exit(0) # exit the script after training. This is just for testing the training loop. Remove this line to continue with the training loop.

# torchrun --standalone --nproc_per_node=4 aae_train_GPU_v2_rotary.py


