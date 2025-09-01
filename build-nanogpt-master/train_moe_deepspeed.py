#%%
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
import time

import deepspeed
import json
import model_moe_deepspeed as model_ds
import eval_log_utils



 #%%
assert torch.cuda.is_available()  ,"This script is designed to run on CUDA devices only. Please ensure you have a compatible GPU."



# This is the configuration for the GPT model. It defines the hyperparameters for the model. The block size is the maximum sequence length, vocab size is the size of the vocabulary, n_layer is the number of transformer blocks, n_head is the number of attention heads, and n_embd is the embedding dimension. 

# Note on effective batch size: tbd

@dataclass
class GPTConfig:
    seq_len: int = 1024
    batch_size: int = 16
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_experts: int = 4
    k: int = 2
    base_lr: float = 6e-4 * 3
    warm_up_steps: int = 300
    target_tokens_per_optimizer_step = 1048576
    # accum_steps: int = 24
    load_balance_scale: float = 0.01

    def __post_init__(self):
        # Distributed training setup
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.master_process = (self.rank == 0)
        
        # Validation
        assert self.n_embd % self.n_head == 0
        assert self.num_experts % self.world_size == 0
        assert self.k <= self.num_experts and self.k > 0
        
        # Compute accumlation steps based on target_tokens_per_optimizer_step, sequence length and world size
        self.accum_steps = self.target_tokens_per_optimizer_step // (self.batch_size * self.seq_len * self.world_size)
        
        # self.effective_batch_size_desired = (
        #     self.batch_size * self.seq_len * self.world_size * self.accum_steps
        # )
        self.training_steps = (10_000_000_000 // self.target_tokens_per_optimizer_step) + 1

# instantiate and check the config
config = GPTConfig()
config.target_tokens_per_optimizer_step


def create_deepspeed_config(config):
    """Create DeepSpeed configuration with proper MoE load balancing"""
    effective_batch_size_deepspeed = config.batch_size * config.accum_steps * config.world_size
    
    ds_config = {
        "train_batch_size": effective_batch_size_deepspeed,
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.accum_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.base_lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0.0,  # Start from 0, not 0.05
                "warmup_num_steps": config.warm_up_steps,
                "total_num_steps": config.training_steps,
                "cos_min_ratio": 0.1
            }
        },
        
        "zero_optimization": {
            "stage": 1,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "reduce_bucket_size": 200000000
        },
        
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }
    
    if config.num_experts > 1:
        ds_config["moe"] = {
            "enabled": True,
            "ep_size": min(config.world_size, config.num_experts),
            "moe_param_group": True,
            "use_residual": False,
            "load_balance_scale": config.load_balance_scale
        }
    
    return ds_config



if config.master_process:
    print(f'\nGPTConfig instantiated with block size: {config.seq_len}, vocab size: {config.vocab_size}, n_layer: {config.n_layer}, n_head: {config.n_head}, n_embd: {config.n_embd}')
    print(f'\ntarget tokens per optimizer step: {config.target_tokens_per_optimizer_step:,}')


# Note that in the initialization of the network in the ffn class, we are multiplying n_embd (the dimensions of the original embeddings) by 4. So for the inner layers, the dimensionality of the model is 768 * 4 


#%%


torch.manual_seed(42) # set the random seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(42) # set the random seed for cuda for reproducibility

def initialize_deepspeed(model, config):
    """Initialize DeepSpeed engine with programmatic config"""
    ds_config = create_deepspeed_config(config)
    
    # Optional: save config for debugging
    if config.master_process:
        with open('deepspeed_config_debug.json', 'w') as f:
            json.dump(ds_config, f, indent=2)
        print(f"\nDeepSpeed config created and saved for debugging\n")
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    if config.master_process:
        print(f"\nDeepSpeed initialized:")
        print(f"  - World size: {config.world_size}")
        print(f"  - Effective batch size: {ds_config['train_batch_size']}")
        print(f"  - Micro batch size per GPU: {ds_config['train_micro_batch_size_per_gpu']}")
        print(f"  - Gradient accumulation steps: {ds_config['gradient_accumulation_steps']}")
        print(f"  - MoE enabled: {ds_config.get('moe', {}).get('enabled', False)}\n")
    
    return model_engine, optimizer, lr_scheduler

# %%

def count_parameters_moe(model, config):
    """Count model parameters for MoE"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # This is approximate since DeepSpeed handles expert parameters internally
    # We'll estimate based on model structure
    expert_params = 0
    shared_params = 0
    
    for name, param in model.named_parameters():
        if 'expert' in name.lower():
            expert_params += param.numel()
        else:
            shared_params += param.numel()
    
    if expert_params == 0:
        # Estimate expert params from MLP size
        expert_params_per_expert = config.n_embd * 4 * config.n_embd * 2  # Rough estimate
        expert_params = expert_params_per_expert * config.num_experts * config.n_layer
        shared_params = total_params - expert_params
    
    expert_params_per_expert = expert_params // (config.num_experts * config.n_layer)
    active_expert_params = expert_params_per_expert * config.k * config.n_layer
    active_params = shared_params + active_expert_params
    
    return {
        'total_params': total_params,
        'shared_params': shared_params,
        'expert_params': expert_params,
        'active_params': active_params,
        'params_per_expert': expert_params_per_expert
    }



# %%
# Instantiate the dataloader and load the data. 



def main():
    import model_moe_deepspeed as model_ds
    # Check CUDA availability
    assert torch.cuda.is_available(), "CUDA required for training"
    
    # FOR SINGLE GPU: Initialize distributed training manually
    if os.environ.get('RANK') is not None:
        print(f'Running in distributed environment. Initializing DeepSpeed')
        deepspeed.init_distributed()
    else:
        print(f'Running in single GPU environment - setting up for DeepSpeed')
        # Set environment variables for single GPU DeepSpeed
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'  
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        # Initialize PyTorch distributed for DeepSpeed (even for single GPU)
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',  # Use 'nccl' for GPU, 'gloo' for CPU
                init_method='env://',
                world_size=1,
                rank=0
            )
    
    # Configuration - recreate after setting env vars
    config = GPTConfig()
    
    if config.master_process:
        print(f'\nGPTConfig instantiated:')
        print(f'  - Block size: {config.seq_len}')
        print(f'  - Vocab size: {config.vocab_size}')
        print(f'  - Layers: {config.n_layer}')
        print(f'  - Heads: {config.n_head}') 
        print(f'  - Embedding dim: {config.n_embd}')
        print(f'  - Experts: {config.num_experts}, k={config.k}')
        print(f'  - target tokens per optimzer step: {config.target_tokens_per_optimizer_step:,}')
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize model
    model = model_ds.CreateMoEDeepSpeed(config)
    
    # Count parameters
    if config.master_process:
        params_info = count_parameters_moe(model, config)
        print(f'\nModel parameters:')
        for key, value in params_info.items():
            print(f"  {key}: {value:,}")


    from dataloader_utils import DataLoaderShardMultiGPU
    # initialize the dataloader for training and validation data. Batch size has to be be customized to fit the gpu being used.
    train_loader = DataLoaderShardMultiGPU(B=config.batch_size, seq_len=config.seq_len, process_rank = config.rank, num_processes=config.world_size, split='train')

    val_loader = DataLoaderShardMultiGPU(B=config.batch_size, seq_len=config.seq_len, process_rank = config.rank, num_processes=config.world_size, split='val')

    assert config.target_tokens_per_optimizer_step % (train_loader.B * train_loader.seq_len * config.world_size) == 0, f"target_tokens_per_optimizer_step {config.target_tokens_per_optimizer_step} is not divisible by batch size {train_loader.B} and sequence length {train_loader.seq_len}"
    
    
    # Initialize DeepSpeed
    model_engine, optimizer, lr_scheduler = initialize_deepspeed(model, config)
    device = model_engine.device
    print(f'\nDeepSpeed model initialized on device: {device}\n')
    
    # Rest of your training code...
    # Initialize data loaders
    train_loader = DataLoaderShardMultiGPU(
        B=config.batch_size, 
        seq_len=config.seq_len, 
        process_rank=config.rank, 
        num_processes=config.world_size, 
        split='train'
    )
    
    val_loader = DataLoaderShardMultiGPU(
        B=config.batch_size, 
        seq_len=config.seq_len, 
        process_rank=config.rank, 
        num_processes=config.world_size, 
        split='val'
    )
    
    # Calculate training steps
    training_steps = 10_000_000_000 // config.target_tokens_per_optimizer_step
    accumulation_steps_desired = config.accum_steps
    
    if config.master_process:
        print(f"\nTraining setup:")
        print(f"  - Training steps for one epoch: {training_steps:,}")
        print(f"  - Accumulation steps: {accumulation_steps_desired}")
    
    # Initialize logging
    log_params = eval_log_utils.LogParamsFilesConfig(
        FSDP=True,  # Keep as True since we're still doing distributed training
        world_size=config.world_size,
        rank=config.rank,
        local_rank=config.local_rank,
        model=model_engine,  # Pass the DeepSpeed engine
        device=device,
        encoder=tiktoken.get_encoding('gpt2'),
        val_loader=val_loader,
        loss_dir="train_loss",
        hella_accu_dir="hella_accuracy", 
        learn_rate_dir='learn_rate_sched',
        train_loss_file="train_loss.csv",
        hella_accu_file="hellaswag_eval.csv",
        lr_file="learning_rate.csv",
        step=0,
        shard_idx=0,
        loss_accum=0.0,
        lr=0.0
    )
    
    # Training loop
    model_engine.train()
    total_tokens_seen = 0
    # print("Optimizer initial LR:", model_engine.optimizer.param_groups[0]['lr'])
    for step in range(training_steps):
        t0 = time.time()
        last_step = (step == training_steps - 1)
        
        # Training step with gradient accumulation
        loss_accum = 0.0
        
        for micro_step in range(accumulation_steps_desired):
            x, y, shard_idx, tokens_abandoned = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model_engine(x, y)
            
            # Scale loss for accumulation
            # loss = loss / accumulation_steps_desired
            loss_accum += loss.detach()
            
            # Backward pass (DeepSpeed handles everything)
            model_engine.backward(loss)
        
        # DeepSpeed step (includes gradient clipping, optimization, and scheduling)
        model_engine.step()
        # Check if scheduler is stepping properly
        if step < 10:  # Debug first 10 steps
            print(f"DEBUG Step {step}:")
            print(f"  Scheduler type: {type(model_engine.lr_scheduler)}")
            print(f"  Scheduler last_epoch: {getattr(model_engine.lr_scheduler, 'last_epoch', 'N/A')}")
            print(f"  Base LR: {config.base_lr}")
            print(f"  Current LR: {model_engine.get_lr()[0]}")
            
            # Force scheduler step if it's not auto-stepping
            if hasattr(model_engine.lr_scheduler, 'step'):
                print(f"  Manual scheduler step...")
        
        # Synchronize and calculate timing
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        
        # Token counting
        tokens_processed_local = train_loader.B * train_loader.seq_len * config.accum_steps
        tokens_processed_total = tokens_processed_local * config.world_size
        tokens_per_sec = tokens_processed_total / dt
        total_tokens_seen += tokens_processed_total
        
        # Update log params and log training progress
        if config.master_process:
            log_params.step = step
            log_params.shard_idx = shard_idx
            # log_params.loss_accum = round(loss_accum.item(), 7)
            log_params.lr = round(model_engine.get_lr()[0], 7)  # DeepSpeed method to get LR
            
            # Log training loss and learning rate
            eval_log_utils.TrainLoss(log_params=log_params).log_training_loss()
            eval_log_utils.LearningRate(log_params=log_params).log_learning_rate()
            
            # Print progress
            print(f"Step {step}, shard_idx: {shard_idx}, Loss: {loss_accum.item():.5f}, "
                  f"LR: {model_engine.get_lr()[0]:.7f}, Time: {dt:.2f}sec, "
                  f"Tokens/sec: {tokens_per_sec:,.0f}")
            print(f"Total tokens seen: {total_tokens_seen:,}")
        
        # Evaluation and logging
        if ((step > 0 and step % 250 == 0) or last_step):
            eval_log_utils.HellaSwag(log_params=log_params).log_print_hella_accuracy()
        
        if step % 250 == 0 and step > 0:
            eval_log_utils.Validation(log_params=log_params).check_validation_loss()
        
        if ((step % 1000 == 0 and step > 0) or last_step):
            eval_log_utils.GenerateSample(log_params=log_params).generate(
                context="Hello, I'm a language model,", sample_max_length=32
            )

if __name__ == "__main__":
    main()