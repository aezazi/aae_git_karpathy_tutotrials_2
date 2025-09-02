#%%
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
import time
import model_moe_fsdp as model_FSDP
import model_moe_fsdp_parallel as model_FSDP_parallel
from model_moe_fsdp import Block

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP_wrap,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy
)
import torch.distributed as dist
import functools


 #%%
assert torch.cuda.is_available()  ,"This script is designed to run on CUDA devices only. Please ensure you have a compatible GPU."

# Check if we are running in FSDP mode. If so, we will initialize the process group and set the device for each process. a simple way to check whether your script is being run under Distributed Data Parallel (FSDP) — specifically when using torchrun with a cuda GPU. Note that you can be in FSDP mode even with a single GPU when using torchrun. 
if os.environ.get('RANK') is not None and os.environ.get('WORLD_SIZE') is not None:
    print(f'Running in distributed environment. Initializing FSDP\n')
    init_process_group(backend='nccl') # initialize the process group 
    
else:
    print(f'Running in a non-distributed environment.\n')


# This is the configuration for the GPT model. It defines the hyperparameters for the model. The block size is the maximum sequence length, vocab size is the size of the vocabulary, n_layer is the number of transformer blocks, n_head is the number of attention heads, and n_embd is the embedding dimension. 

# Note on effective batch size: tbd

@dataclass
class GPTConfig:
    seq_len: int = 2048 # max sequence length
    # setting vocab size to 50304 rather than 50257 (the size of the gpt2 vocab) because this is a much more efficient number (divisible by many powers of 2) for gpu kernels and computations. The extra tokens are just padding tokens that are not used in the model. The model will learn to ignore them. this is a tradeoff between memory and performance. 
    model_expert_parallelization = False # choose whether to run the model with just fsdp or the model with fsdp and expert parallelization
    batch_size = 8
    # effective_batch_size_multiplier = 8
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    base_lr = 6e-4 * 3
    warm_up_steps = 300
    target_tokens_per_optimizer_step = 1048576
    num_experts = 16
    load_balance_scale = 0.01
    k = 2
    print_token_routing = True

    def __post_init__(self):
        # Note that LOCAL_RANK is the rank of the process on one given machine (when using multiple machine), while RANK is the rank of the process across all machines (when using multiple gpus on multiple machines). When using a setup with just one machine, LOCAL_RANK and RANK are the same.
        
        self.FSDP = dist.is_initialized()
        self.world_size = dist.get_world_size() if self.FSDP else 1

        assert self.n_embd % self.n_head == 0 

        assert self.num_experts % self.world_size == 0, f"num_experts ({self.num_experts}) must be divisible by world_size ({self.world_size})"

        assert self.k <= self.num_experts and self.k > 0, f"k must be at least 1 and less than or equal to num_experts {self.num_experts} you have k={self.k}"

        # Compute accumlation steps based on target_tokens_per_optimizer_step, sequence length and world size
        self.accum_steps = self.target_tokens_per_optimizer_step // (self.batch_size * self.seq_len * self.world_size)

        self.training_steps = (10_000_000_000 // self.target_tokens_per_optimizer_step) + 1

        # self.effective_batch_size_desired = self.batch_size * self.seq_len * self.world_size * self.effective_batch_size_multiplier

        self.experts_per_gpu = self.num_experts // self.world_size
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank =  int(os.environ['LOCAL_RANK']) # get the local rank of the current process
        self.master_process = (self.rank == 0)

# instantiate and check the config
config = GPTConfig()

if config.master_process:
    print(f'\nGPTConfig instantiated with block size: {config.seq_len}, vocab size: {config.vocab_size}, n_layer: {config.n_layer}, n_head: {config.n_head}, n_embd: {config.n_embd}')
    # print(f'\neffective batch size desired: {config.effective_batch_size_desired:,}')


# Note that in the initialization of the network in the ffn class, we are multiplying n_embd (the dimensions of the original embeddings) by 4. So for the inner layers, the dimensionality of the model is 768 * 4 


#%%
# FSDP setup

if config.FSDP:
    # set the device to the local rank of the current process
    device = f'cuda:{config.local_rank}' 
    torch.cuda.set_device(device) # set the device for the current process

    print(f'\nFSDP initialized on device: {device}, rank: {config.rank}, local rank: {config.local_rank}, world size: {config.world_size}\n')

# if not using FSDP
else: 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        config.rank = 0
        config.local_rank = 0
        config.world_size = 1
        master_process = True

torch.manual_seed(42) # set the random seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(42) # set the random seed for cuda for reproducibility


# %%
#Instantiate the model based on whether expert paralleliztion was chosen in config
# if cuda is available, use torch.compile to optimize the model for training on GPUs. This is a performance optimization that allows for more efficient training on GPUs. It uses the PyTorch JIT compiler to optimize the model for the specific hardware and software configuration. This is done to improve performance and reduce memory usage. we use bfloat16 precision for the forward pass and use torch.compile. See Karpathy's tutorial at 1:24:00 and 1:49:00 for details. NOTE   that compile may not play well with FSDP and especially not well with manual expert parallelization communications . So will have to experiment.
if config.model_expert_parallelization:
    model = model_FSDP_parallel.CreateMoEParalell(config=config)
    use_compile = False # set to True to use torch.compile
else:
    model = model_FSDP.CreateMoE(config=config)
    use_compile = True # set to True to use torch.compile

# compute number of model parameters
# def count_parameters(model, config=None):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_moe(model, config=config):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count expert parameters
    expert_params = 0
    shared_params = 0
    
    for name, param in model.named_parameters():
        if 'expert' in name.lower() or 'moe' in name.lower():
            expert_params += param.numel()
        else:
            shared_params += param.numel()
    
    # Active parameters per forward pass
    expert_params_per_expert = expert_params // config.num_experts
    active_expert_params = expert_params_per_expert * config.k
    active_params = shared_params + active_expert_params
    
    return {
        'total_params': total_params,
        'shared_params': shared_params, 
        'expert_params': expert_params,
        'active_params': active_params,
        'params_per_expert': expert_params_per_expert
    }


if config.master_process:
    params_counted = count_parameters_moe(model)
    print(f'\n')
    for key, value in params_counted.items():
        print(f"{key}: {value:,}")
    # print(params_counted)
    print(f'\n')


# torch.set_float32_matmul_precision('high')
model.to(device)

model = torch.compile(model) if use_compile else model 


if config.model_expert_parallelization:
    from model_moe_fsdp_parallel import Block, MoELayerParallel, ExpertMoESwiglu, TopKGateParallel
    def moe_aware_auto_wrap_policy(module, recurse, nonwrapped_numel):
        """
        Custom FSDP wrap policy that treats MoE components specially.
        
        Key principles:
        1. MoE experts should NOT be sharded across GPUs (they need to be locally available)
        2. The gate can be replicated or wrapped as a unit
        3. Regular transformer components (attention, etc.) can be sharded normally
        """
        
        # The entire MoE layer should be wrapped as a unit to preserve expert locality
        if isinstance(module, MoELayerParallel):
            # print(f"[FSDP] Wrapping entire MoE layer as atomic unit")
            return True
        
        # For regular transformer components, use the standard policy
        return transformer_auto_wrap_policy(
            module, 
            recurse, 
            nonwrapped_numel,
            transformer_layer_cls={Block}
        )

    transformer_wrapper_policy = moe_aware_auto_wrap_policy
else:
        
    # With FSDP, we can wrap different parts of the model. Here I am following a strategy presented in a pytorch tutorial to wrap the transformer block. It's possibel to separately wrap the Moe layer. Will experiment when I get this working.
    transformer_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls = {Block} # transformer layer class as per pytorch tutorial video
    )

# FSDP also allows us to define a mixed prescision policy. Here, I am just using bf16 for everything, but we can use hybrid. refer to this tutorial for more good info and nuances https://www.youtube.com/watch?v=-caN92JtKqA&list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT&index=4
precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16, # param precision
            reduce_dtype=torch.bfloat16, # gradient communication precision
            buffer_dtype=torch.bfloat16 # buffer precision
            )

# wrap model per wrapper policy
model = FSDP_wrap(model,
            auto_wrap_policy=transformer_wrapper_policy,
            mixed_precision=precision_policy,
        
            # reccommendation and other good info from tutorial: https://www.youtube.com/watch?v=sDM56HOziE4&list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT&index=8
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,

            # note that ShardingStrategy.NO_SHARD is the equivalent of having the model run in DDP mode
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            
            device_id=torch.cuda.current_device(),
            forward_prefetch=True
            )

print(f"\n[FSDP] Rank {config.rank}: Model wrapping complete\n")

#%%
# Instantiate the optimizer.
from aae_utils import ConfigureOptimizer

# Note that in the Karpathy tutotrial he uses DDP  and not FSDP. The optimizer initialization when using DDP is slightly different in that you have to use  the "raw model" parameters beofere wrapping with DDP for optimizer initialization. With FSDP, you can just pass the wrapped model.Refer to the tutorial.
optimizer = ConfigureOptimizer(model).create_optimizer(weight_decay=0.1, learning_rate = config.base_lr, device_type=device)

if config.FSDP:
    print(f'\nOptimizer initialized on GPU rank {config.rank}, device {device}')


# %%
# Instantiate the dataloader and load the data. 
from dataloader_utils import DataLoaderShardMultiGPU


# initialize the dataloader for training and validation data. Batch size has to be be customized to fit the gpu being used.
train_loader = DataLoaderShardMultiGPU(B=config.batch_size, seq_len=config.seq_len, process_rank = config.rank, num_processes=config.world_size, split='train')

val_loader = DataLoaderShardMultiGPU(B=config.batch_size, seq_len=config.seq_len, process_rank = config.rank, num_processes=config.world_size, split='val')

assert config.target_tokens_per_optimizer_step % (train_loader.B * train_loader.seq_len * config.world_size) == 0, f"effective batch size {config.effective_batch_size_desired} is not divisible by batch size {train_loader.B} and sequence length {train_loader.seq_len}"

# this is the desired number of micro steps to accumulate gradients over. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU. In my implementation this is equal to effective_batch_size_multiplier. Karpathy starts with effective batch_size desired to match gpt2 implementation and then computes accumulation steps to make sure the formula below works with batch_size
# accumulation_steps_desired = config.effective_batch_size_desired // (train_loader.B * train_loader.seq_len * config.world_size) 

if config.master_process:
    print(f"\neffective batch size is the same as target tokens per optimizer step: {config.target_tokens_per_optimizer_step}")
    print(f"accumulation steps: {config.accum_steps}\n")

#%%
# Instantiate the learning rate scheduler 
# NOTE: I moved the code for the scheduler to a separate aae_utils.py file.
from aae_utils import CosineLearingRateScheduler


# compute training steps for 1 epoc. compute number of steps for one pass over our training dataset of 10B tokens
# training_steps = 10_000_000_000 // config.effective_batch_size_desired 
print(f'\ntraining steps for one epoc: {config.training_steps:,}\n')

# define the scheduler parameters
# the number of iterations over which lr is reduced to the minimum
T_max = config.training_steps 

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
print(f'\nScheduler initialized on GPU rank {config.rank}, of {config.world_size}\n')

#%%
# create log files, loggers, and evaluators to store training loss, learning rate, validation loss, hellaswag eval results, and generate sample text.
import eval_log_utils as eval_log_utils
log_params = eval_log_utils.LogParamsFilesConfig(
    FSDP = config.FSDP,
    world_size = config.world_size,
    rank = config.rank,
    local_rank = config.local_rank,
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

for step in range(config.training_steps):
    t0 = time.time()
    last_step = (step == config.training_steps - 1)

    # Main training loop
    optimizer.zero_grad()
    loss_accum  = 0.0
    micro_steps = config.accum_steps # set the number of mirco steps to accumulate gradients over
    for micro_step in range(micro_steps):
        # this is a gradient accumulation step. We accumulate gradients over desired accumalation steps before updating the weights. This is done to reduce the number of weight updates and improve training stability. It is also done to reduce the memory usage on the GPU. 
        x, y, shard_idx, tokens_abandoned = train_loader.next_batch()
        x, y = x.to(device), y.to(device) # move the data to the device. 

        # By default, FSDP synchronizes the loss from each process after each micro step by taking an average of all the processes and making that average the loss for all the processes for that step. Its very inefficient to do this at each micro_step. So we want to only synchronize gradients among all the processes on the last micro step. See Karpathy's video tutorial at 2:57:00 for more details. The code below sets the require_backward_grad_sync attribute of the model to True only on the last micro step. 
        if config.FSDP:
            model.require_backward_grad_sync = (micro_step == micro_steps - 1) 

        # we use autocast to use bfloat16 precision for the forward pass. This is a performance optimization for training on GPUs. The device must be cuda.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss, top_k_all = model(x, y)

        # This is a check to make sure the top_k gate is distributing tokens evenly bewtween the experts. Code below checks every block. This is Claude generated code
      
        with torch.no_grad():
            for layer_idx, top_k_global_ids in enumerate(top_k_all):
                # Get local expert usage counts for this GPU
                local_counts = torch.bincount(top_k_global_ids, minlength=config.num_experts)
                
                if config.FSDP:
                    # Aggregate counts across all GPUs to get true expert utilization
                    global_counts = local_counts.clone()
                    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
                    accum_topk_expert_count[layer_idx] += global_counts
                else:
                    # Single GPU case
                    accum_topk_expert_count[layer_idx] += local_counts
        

        # divide the loss by the number of micro steps to get the average loss of the accumulated micro steps
        # Divide by accumulation steps (so gradient is averaged across micro-batches)
        microbatch_loss = loss / micro_steps 
        
        # Look at Pytorch documentation for more details on tensor.detach() vs. tensor.item()
        loss_accum += microbatch_loss.detach() 
        microbatch_loss.backward()


    if config.FSDP:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip the gradients to prevent exploding gradients
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.set_lr(step)

    # synchronize the device to make sure all operations are complete before measuring time
    torch.cuda.synchronize()
    
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed_local = train_loader.B * train_loader.seq_len * micro_steps
    if config.FSDP:
        # In FSDP, each GPU processes different data, so total tokens is sum across GPUs
        tokens_processed_total = tokens_processed_local * config.world_size
    else:
        # Single GPU case
        tokens_processed_total = tokens_processed_local

    tokens_per_sec = tokens_processed_total / dt
    total_tokens_seen += tokens_processed_total
    
    
    
    # update log_params, log training loss and learning rate to file, print processing stats.
    if config.master_process:
        # update log_params
        log_params.step = step
        log_params.shard_idx = shard_idx
        log_params.loss_accum = round(loss_accum.item(), 7)
        log_params.lr = round(optimizer.param_groups[0]['lr'], 7)

        #log training loss and learning rate to file
        eval_log_utils.TrainLoss(log_params=log_params).log_training_loss()
        eval_log_utils.LearningRate(log_params=log_params).log_learning_rate()

        # print processing stats
        print(f"Step {step},  shard_idx: {shard_idx},  Loss: {loss_accum.item():.5f},  LR: {optimizer.param_groups[0]['lr']:.7f},  norm: {norm:.4f}, Time: {dt:.2f}sec,  Tokens/sec: {tokens_per_sec:,.0f} \ntotal tokens seen: {total_tokens_seen:,}")

    # expert utilization tracking code from Claude
    if config.print_token_routing and step % 1000 == 0:
        if config.master_process:  # Only print from master process
            print(f'\n=== Expert Utilization Statistics (Step {step}) ===')
            print(f'Total tokens processed across all GPUs: {total_tokens_seen:,}')
            
            for i, counts in enumerate(accum_topk_expert_count):
                print(f"\nLayer {i} - Raw counts: {counts.tolist()}")
                
                # Calculate normalized usage (should sum to k * total_tokens for top-k=2)
                normalized = counts.float() / total_tokens_seen
                print(f"Layer {i} - Normalized: {normalized.tolist()}")
                
                # Calculate expert usage balance (coefficient of variation)
                mean_usage = normalized.mean()
                std_usage = normalized.std()
                cv = (std_usage / mean_usage).item() if mean_usage > 0 else float('inf')
                print(f"Layer {i} - Balance (CV): {cv:.4f} (lower is more balanced)")
                
                # Show which experts are over/under utilized
                expected_usage = config.k / config.num_experts  # Expected usage for balanced experts
                over_utilized = (normalized > expected_usage * 1.5).sum().item()
                under_utilized = (normalized < expected_usage * 0.5).sum().item()
                print(f"Layer {i} - Over-utilized experts: {over_utilized}, Under-utilized: {under_utilized}")
            
            print(f'================================\n')

    # every x steps evaluate, print, and log hellaswag.
    if ((step > 0 and step % 250 == 0) or last_step):
        eval_log_utils.HellaSwag(log_params=log_params).log_print_hella_accuracy()

    # Every x steps, put the model in validation mode and use the validation dataset to compute loss. This is to help us catch any over fitting issues. 
    if step % 250 == 0 and step > 0:
        eval_log_utils.Validation(log_params=log_params).check_validation_loss()
    
    # every x steps generate from the model.
    if ((step % 1000 == 0 and step > 0) or last_step):
        eval_log_utils.GenerateSample(log_params=log_params).generate(context="Hello, I'm a language model,", sample_max_length=32)

if config.FSDP:
    destroy_process_group()

import sys; sys.exit(0) # exit the script after training. This is just for testing the training loop. Remove this line to continue with the training loop.

# torchrun --standalone --nproc_per_node=1 train_moe_fsdp.py


