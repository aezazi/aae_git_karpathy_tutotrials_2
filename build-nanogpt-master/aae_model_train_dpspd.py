#%%
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import deepspeed
# from hellaswag import render_example, iterate_examples
import tiktoken
import time
import aae_model_rotary_moe_dpspd as model_moe


#%%
# assert torch.cuda.is_available()  ,"This script is designed to run on CUDA devices only. Please ensure you have a compatible GPU."
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# %%
# This is the configuration for the GPT model. It defines the hyperparameters for the model. The block size is the maximum sequence length, vocab size is the size of the vocabulary, n_layer is the number of transformer blocks, n_head is the number of attention heads, and n_embd is the embedding dimension. 

@dataclass
class GPTConfig:
    seq_len: int = 1024 # max sequence length
    # setting vocab size to 50304 rather than 50257 (the size of the gpt2 vocab) because this is a much more efficient number (divisible by many powers of 2) for gpu kernels and computations. The extra tokens are just padding tokens that are not used in the model. The model will learn to ignore them. this is a tradeoff between memory and performance. 
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    num_experts = 4
    world_size = gpu_count
    ep_size = world_size
    assert ep_size <= num_experts, "ep_size must be less than or equal to num_experts"
    assert num_experts % ep_size == 0, "num_experts must be divisible by ep_size"
    k = 2
    noisy_std = 1.0
    effective_batch_size_desired_tokens = 524288 # 2^19 ~ .5M to match the original GPT-2 paper.
    seq_len = 1024
    effective_batch_size = effective_batch_size_desired_tokens // seq_len 
    mini_batch_size = 16 # this is the batch size per gpu. The effective batch size is the product of the mini-batch size and the number of gpus.
    n_embd: int = 768
    base_lr: float = 6e-4 * 4
    max_lr: float = base_lr # max learning rate
    min_lr_ratio: float =  0.1
    warm_up_steps: int = 300
    train_steps: int = 19703 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
# instantiate and check the config
config = GPTConfig()


#%%
# create the deepspeed config.
ds_config = {
    # === Training batch settings ===
    "train_batch_size": config.effective_batch_size,           # total global batch size
    "train_micro_batch_size_per_gpu": config.mini_batch_size,  # per-GPU batch size
    # "gradient_accumulation_steps": 4,

    # === Mixed precision ===
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },

    # === ZeRO optimization ===
    "zero_optimization": {
        "stage": 2,
        "offload_param": {
            "device": "none"
        },
        "allgather_partitions": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },

    # === Optimizer ===
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    # === LR Scheduler ===
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 1000
        }
    },

    # === MoE-specific configuration ===
    "moe": {
        "enabled": True,
        "num_experts": 16,              # total experts per MoE layer
        "top_k": 2,                     # number of experts to activate per token
        "capacity_factor": 1.25,        # controls expert load capacity
        "min_capacity": 4,              # minimum tokens per expert
        "noisy_gate_policy": "RSample", # or None, 'Jitter'
        "use_residual": False,          # enable residual MoE connections
        "aux_loss_coeff": 0.01          # auxiliary loss weight for balancing
    },

    # ✅ Crucial for auto MoE param grouping
    "moe_param_group": True,

    # === Misc ===
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": False
}

print(f"\nusing device: {device}")

print(f'\nGPTConfig instantiated with block size: {config.seq_len}, vocab size: {config.vocab_size}, n_layer: {config.n_layer}, n_head: {config.n_head}, n_embd: {config.n_embd}')


torch.manual_seed(42) # set the random seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(42) # set the random seed for cuda for reproducibility


# %%
#Instantiate the model. Note that we do not use torch.compile since it's not comaptible with DeepSpeed MoE layer

model = model_moe.CreateDeepSpeedMoE(config=config)

# compute number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {count_parameters(model):,}\n")


# torch.set_float32_matmul_precision('high')

#%%
# import deepspeed.moe.layer as moe_layer
# import deepspeed.moe.utils

moe_params = []
non_moe_params = []

# for name, param in model.named_parameters():
#     if "deepspeed_moe_experts" in name or "experts" in name:   # adjust as needed
#         moe_params.append(param)
#     else:
#         non_moe_params.append(param)

# param_groups = [
#     {"params": non_moe_params},  # regular params
#     {"params": moe_params, "moe": True, "name": "expert"}  # ✅ name required
# ]

# for name, _ in model.named_parameters():
#     print(name)

# len(moe_params)

#%%
# Instantiate the optimizer.
# NOTE: I moved the code for optimizer configuration to a separate file called aae_utils.py.
from aae_utils import ConfigureOptimizerDS




#%%
# Initialize DeepSpeed engine. This is where the model, optimizer, and scheduler are wrapped in a DeepSpeed engine.
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    # optimizer=optimizer,    # manually created AdamW
    # lr_scheduler=scheduler, # manually created WarmupCosineLR
    model_parameters=model.parameters(),
    config=ds_config   # only contains ZeRO/MoE/etc. (no scheduler section)
)

print(f'I am here')
# %%
# Instantiate the dataloader and load the data. 
from aae_dataloader_utils import DataLoaderShardMultiGPU

# initialize the dataloader for training and validation data. Batch size has to be be customized to fit the gpu being used.
B = 32 # batch size

train_loader = DataLoaderShardMultiGPU(B=B, seq_len=config.seq_len, process_rank = ddp_rank, num_processes=ddp_world_size, split='train')

val_loader = DataLoaderShardMultiGPU(B=B, seq_len=config.seq_len, process_rank = ddp_rank, num_processes=ddp_world_size, split='val')

# we want to match the batch size of 0.5M used in the GPT2. Our GPUs can't handle that. So we will use a smaller batch size and accumulate gradients over multiple steps to get the same effect. See the training loop below for details on implementing gradient accumulation.
effective_batch_size_desired =524288 # 2^19 ~ .5M to match the original GPT-2 paper. 


assert effective_batch_size_desired % (train_loader.B * train_loader.seq_len * ddp_world_size) == 0, f"effective batch size {effective_batch_size_desired} is not divisible by batch size {train_loader.B} and sequence length {train_loader.T}"

# this is the desired number of micro steps to accumulate gradients over. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU.
accumulation_steps_desired = effective_batch_size_desired // (train_loader.B * train_loader.seq_len * ddp_world_size) 

if master_process:
    print(f"\neffective batch size desired: {effective_batch_size_desired}")
    print(f"accumulation steps desired: {accumulation_steps_desired}")



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

# deepspeed --num_gpus=1  aae_model_train_dpspd.py


