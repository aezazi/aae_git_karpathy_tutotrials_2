#%%
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
import time
import aae_model_moe_ddp as model_rotary_moe


# #%%
# assert torch.cuda.is_available()  ,"This script is designed to run on CUDA devices only. Please ensure you have a compatible GPU."

# %%
# This is the configuration for the GPT model. It defines the hyperparameters for the model. The block size is the maximum sequence length, vocab size is the size of the vocabulary, n_layer is the number of transformer blocks, n_head is the number of attention heads, and n_embd is the embedding dimension. 


@dataclass
class GPTConfig:
    seq_len: int = 1024 # max sequence length
    # setting vocab size to 50304 rather than 50257 (the size of the gpt2 vocab) because this is a much more efficient number (divisible by many powers of 2) for gpu kernels and computations. The extra tokens are just padding tokens that are not used in the model. The model will learn to ignore them. this is a tradeoff between memory and performance. 
    batch_size = 32
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    base_lr = 6e-4 * 3
    warm_up_steps = 300
    num_experts = 8
    k = 2
    print_token_routing = True

# instantiate and check the config
config = GPTConfig()

print(f'\nGPTConfig instantiated with block size: {config.seq_len}, vocab size: {config.vocab_size}, n_layer: {config.n_layer}, n_head: {config.n_head}, n_embd: {config.n_embd}')

"""
Note that in the initialization of the network in the ffn class, we are multiplying n_embd (the dimensions of the original embeddings) by 4. So for the inner layers, the dimensionality of the model is 768 * 4 
"""



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
    ddp_rank = dist.get_rank() # get the rank of the current process
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # get the local rank of the current process
    ddp_world_size = dist.get_world_size() # get the total number of processes
    
    # set the device to the local rank of the current process
    device = f'cuda:{ddp_local_rank}' 
    torch.cuda.set_device(device) # set the device for the current process

    # the master process will perform logging and saving checkpoints.
    master_process = (ddp_rank == 0)

    print(f'\nDDP initialized on device: {device}, rank: {ddp_rank}, local rank: {ddp_local_rank}, world size: {ddp_world_size}')

# if not using DDP, just use the next best available option
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

model = model_rotary_moe.CreateMoE(config=config)

# compute number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {count_parameters(model):,}\n")

torch.set_float32_matmul_precision('high')
model.to(device)
use_compile = True # set to True to use torch.compile
model = torch.compile(model) if use_compile else model 

# wrap the model in DDP if using DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    print(f'\nModel wrapped in DDP on device: {device}')


use_compile = True # set to True to use torch.compile
model = torch.compile(model) if use_compile else model 
# get the raw model from the DDP wrapper. DDP is just a wrapper around your model that adds gradient syncing.The actual parameters live inside model.module. So thats what we will need to pass to the optimizer
raw_model = model.module if ddp else model 


#%%
# Instantiate the optimizer.
# NOTE: I moved the code for optimizer configuration to a separate file called aae_utils.py.
from aae_utils import ConfigureOptimizer



# Note that we are using the raw model here, not the DDP wrapped model. This is because the DDP wrapper does not have the optimizer parameters. The raw model is the actual model that we want to optimize.
optimizer = ConfigureOptimizer(raw_model).create_optimizer(weight_decay=0.1, learning_rate = config.base_lr, device_type=device)

if ddp:
    print(f'\nOptimizer initialized on GPU rank {ddp_rank}, device {device}')


# %%
# Instantiate the dataloader and load the data. 
from aae_dataloader_utils import DataLoaderShardMultiGPU

# initialize the dataloader for training and validation data. Batch size has to be be customized to fit the gpu being used.

train_loader = DataLoaderShardMultiGPU(B=config.batch_size, seq_len=config.seq_len, process_rank = ddp_rank, num_processes=ddp_world_size, split='train')

val_loader = DataLoaderShardMultiGPU(B=config.batch_size, seq_len=config.seq_len, process_rank = ddp_rank, num_processes=ddp_world_size, split='val')

# we want to match the batch size of 0.5M used in the GPT2. Our GPUs can't handle that. So we will use a smaller batch size and accumulate gradients over multiple steps to get the same effect. See the training loop below for details on implementing gradient accumulation.
effective_batch_size_desired =524288 # 2^19 ~ .5M to match the original GPT-2 paper. 


assert effective_batch_size_desired % (train_loader.B * train_loader.seq_len * ddp_world_size) == 0, f"effective batch size {effective_batch_size_desired} is not divisible by batch size {train_loader.B} and sequence length {train_loader.T}"

# this is the desired number of micro steps to accumulate gradients over. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU.
accumulation_steps_desired = effective_batch_size_desired // (train_loader.B * train_loader.seq_len * ddp_world_size) 

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

max_lr = config.base_lr # max learning rate
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
    fsdp_ddp = ddp,
    world_size = ddp_world_size,
    rank = ddp_rank,
    local_rank = ddp_local_rank,
    model = model,
    device = device,
    encoder = tiktoken.get_encoding('gpt2'),
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

# counter and container for tracking how many tokens are assigned to each expert by transformer layer
total_tokens_seen = 0
accum_topk_expert_count = [torch.zeros(config.num_experts, device=device, dtype=torch.long) for _ in range(config.n_layer)]

for step in range(training_steps):
    t0 = time.time()
    last_step = (step == training_steps - 1)

    # Main training loop
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
            logits, loss, top_k_all = model(x, y)
        
        
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
    tokens_processed = train_loader.B * train_loader.seq_len * micro_steps * ddp_world_size
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

    if config.print_token_routing and step % 100 == 0:
        print(f'\n')
        for i, c in enumerate(accum_topk_expert_count):
            print(f"Layer {i}: {c.tolist()}")
        print(f'\n')
        for i, c in enumerate(accum_topk_expert_count):
            print(f"Layer {i} normalized: {(c / total_tokens_seen)}")
        print(f'\n')

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

# torchrun --standalone --nproc_per_node=1 aae_model_train_ddp.py


