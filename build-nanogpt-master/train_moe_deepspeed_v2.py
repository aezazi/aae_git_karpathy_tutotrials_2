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
def main():
    # Check CUDA availability
    assert torch.cuda.is_available(), "CUDA required for training"
    
    # FOR SINGLE GPU: Initialize distributed training manually
    if os.environ.get('RANK') is not None:
        deepspeed.init_distributed()
        print(f'Running in distributed environment. Initializing DeepSpeed')
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



# This is the configuration for the GPT model. It defines the hyperparameters for the model. The block size is the maximum sequence length, vocab size is the size of the vocabulary, n_layer is the number of transformer blocks, n_head is the number of attention heads, and n_embd is the embedding dimension. 

# Note on effective batch size: tbd

@dataclass
class GPTConfig:
    seq_len: int = 1024
    batch_size: int = 8
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_experts: int = 4
    k: int = 2
    base_lr: float = 1e-3
    warm_up_steps: int = 100
    target_tokens_per_optimizer_step = 1048576 // 2
    load_balance_scale: float = 0.5

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
        
        # Compute accumulation steps
        self.accum_steps = self.target_tokens_per_optimizer_step // (self.batch_size * self.seq_len * self.world_size)
        self.training_steps = (10_000_000_000 // self.target_tokens_per_optimizer_step) + 1

def create_deepspeed_config(config):
    """Create DeepSpeed configuration with proper MoE load balancing"""
    effective_batch_size_deepspeed = config.batch_size * config.accum_steps * config.world_size
    
    ds_config = {
        "train_batch_size": effective_batch_size_deepspeed,
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.accum_steps,
        "gradient_clipping": 1.0,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.base_lr,
                "weight_decay": 0.1,
                "torch_adam": True,
                "adam_w_mode": True
            }
        },

        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0.0,
                "warmup_num_steps": config.warm_up_steps,
                "total_num_steps": config.training_steps,
                "cos_min_ratio": 0.1,
                "warmup_type": "linear"
            }
        },

        "zero_optimization": {
            "stage": 0,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True
        },
        
        "bf16": {"enabled": False},
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }
    
    if config.num_experts > 1:
        ds_config["moe"] = {
            "enabled": True,
            "ep_size": config.num_experts,
            "moe_param_group": False,
            "use_residual": False,
            "load_balance_scale": config.load_balance_scale
        }
    
    return ds_config

def initialize_deepspeed(model, config):
    """Initialize DeepSpeed engine with programmatic config"""
    ds_config = create_deepspeed_config(config)
    
    if config.master_process:
        with open('deepspeed_config_debug.json', 'w') as f:
            json.dump(ds_config, f, indent=2)
        print(f"\nDeepSpeed config created and saved for debugging\n")
    
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

def count_parameters_moe(model, config):
    """Count model parameters for MoE"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    expert_params = 0
    shared_params = 0
    
    for name, param in model.named_parameters():
        if 'expert' in name.lower():
            expert_params += param.numel()
        else:
            shared_params += param.numel()
    
    if expert_params == 0:
        expert_params_per_expert = config.n_embd * 4 * config.n_embd * 2
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

def main():
    # Check CUDA availability
    assert torch.cuda.is_available(), "CUDA required for training"
    
    # FOR SINGLE GPU: Initialize distributed training manually
    if os.environ.get('RANK') is not None:
        print(f'Running in distributed environment. Initializing DeepSpeed')
        deepspeed.init_distributed()
    else:
        print(f'Running in single GPU environment - setting up for DeepSpeed')
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'  
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=1,
                rank=0
            )

    # Initialize config
    config = GPTConfig()

    if config.master_process:
        print(f'\nGPTConfig instantiated:')
        print(f'  - Block size: {config.seq_len}')
        print(f'  - Vocab size: {config.vocab_size}')
        print(f'  - Layers: {config.n_layer}')
        print(f'  - Heads: {config.n_head}') 
        print(f'  - Embedding dim: {config.n_embd}')
        print(f'  - Experts: {config.num_experts}, k={config.k}')
        print(f'  - target tokens per optimizer step: {config.target_tokens_per_optimizer_step:,}')
    
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

    # Initialize DeepSpeed
    model_engine, optimizer, lr_scheduler = initialize_deepspeed(model, config)
    device = model_engine.device
    print(f'\nDeepSpeed model initialized on device: {device}\n')

    # After initialize_deepspeed(...)
    if config.master_process:
        print("Engine device:", model_engine.device)
        print("Model engine has optimizer?:", hasattr(model_engine, "optimizer") and model_engine.optimizer is not None)
        try:
            print("model_engine.get_lr():", model_engine.get_lr())
        except Exception:
            print("model_engine.get_lr() not available")
        # show optimizer param groups and lr
        try:
            opt = model_engine.optimizer
            print("Num param groups:", len(opt.param_groups))
            print("Param group lrs (first 10 groups if many):", [g.get("lr", None) for g in opt.param_groups[:10]])
            print("Sizes of first few param groups:", [len(g['params']) for g in opt.param_groups[:10]])
        except Exception as e:
            print("Could not inspect optimizer param_groups:", e)

    if config.master_process:
        print(f"\nTraining setup:")
        print(f"  - Training steps: {config.training_steps:,}")
        print(f"  - Accumulation steps: {config.accum_steps}")

    # Initialize logging
    log_params = eval_log_utils.LogParamsFilesConfig(
        FSDP=True,
        world_size=config.world_size,
        rank=config.rank,
        local_rank=config.local_rank,
        model=model_engine,
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
    
    for step in range(config.training_steps):

        # --- CHECKSUM BEFORE STEP (place at start of outer step) ---
        if config.master_process:
            def param_checksum(mod):
                s = 0.0
                for p in mod.parameters():
                    if p.requires_grad:
                        # convert to float to avoid dtype issues
                        s += float(p.data.float().abs().mean())
                return s

            checksum_before = param_checksum(model_engine.module)
            print(f"[DEBUG] Step {step} checksum_before: {checksum_before:.6e}")
        # ------------------------------------------------------------


        t0 = time.time()
        last_step = (step == config.training_steps - 1)
        
        # Training step with gradient accumulation
        ce_loss_accum = 0.0
        aux_loss_accum = 0.0
        expert_counts_total = None
        
        for micro_step in range(config.accum_steps):
            x, y, shard_idx, tokens_abandoned = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # DEBUGGING: Check input data
            # if step < 5 and micro_step == 0 and config.master_process:
                # print(f"Step {step}: Input stats - x: min={x.min()}, max={x.max()}, y: min={y.min()}, max={y.max()}")
            
            # Forward pass
            ce_loss, aux_loss_sum, exp_counts_sum, logits = model_engine(x, y)
            # if step < 3 and config.master_process:
            #     print("DEBUG requires_grad: ce_loss:", ce_loss.requires_grad,
            #         "aux_loss_sum:", aux_loss_sum.requires_grad)
            
            
            # DEBUGGING: Check for NaN/Inf
            # if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            #     print(f"WARNING: CE Loss is NaN/Inf at step {step}, micro_step {micro_step}")
            #     continue
                
            if torch.isnan(aux_loss_sum) or torch.isinf(aux_loss_sum):
                print(f"WARNING: Aux Loss is NaN/Inf at step {step}, micro_step {micro_step}")
                aux_loss_sum = torch.zeros_like(aux_loss_sum)
            
            # Combine losses
            combined_loss = ce_loss + (aux_loss_sum * config.load_balance_scale)
            
            # Scale loss for accumulation
            microbatch_loss = combined_loss / config.accum_steps
            
            # Accumulate unscaled losses for logging
            ce_loss_accum += ce_loss.detach()
            aux_loss_accum += aux_loss_sum.detach()
            
            # Track expert usage
            if exp_counts_sum is not None:
                if expert_counts_total is None:
                    expert_counts_total = exp_counts_sum.clone()
                else:
                    expert_counts_total += exp_counts_sum
            
            # Backward pass
            model_engine.backward(microbatch_loss)

            # # --- DEBUG: check grads immediately after backward (inside micro-step loop) ---
            # if step < 5 and config.master_process:
            #     # basic booleans
            #     print(f"[DEBUG] step {step} micro_step {micro_step}: microbatch_loss.requires_grad = {microbatch_loss.requires_grad}")
            #     print(f"[DEBUG] ce_loss.grad_fn = {ce_loss.grad_fn}, aux_loss.grad_fn = {aux_loss_sum.grad_fn}")

            #     any_grad = False
            #     # check a few representative params (head, attn, first expert, lm_head)
            #     checks = [
            #         "lm_head.weight",
            #         "transformer.h.0.attn.c_attn.weight",
            #         "transformer.h.0.moe.deepspeed_moe.experts.deepspeed_experts.0.linear_1.weight",
            #         "transformer.h.0.moe.deepspeed_moe.experts.deepspeed_experts.0.c_proj.weight"
            #     ]
            #     for name, p in model_engine.module.named_parameters():
            #         if name in checks:
            #             print(f"[DEBUG] param {name}: requires_grad={p.requires_grad}, grad is None? {p.grad is None}")
            #             if p.grad is not None:
            #                 print(f"    grad norm: {p.grad.norm().item():.6e}")
            #                 any_grad = True

            #     # if none of the representative params have grads, try a broad scan for any grads
            #     if not any_grad:
            #         any_present = False
            #         for name, p in model_engine.module.named_parameters():
            #             if p.grad is not None:
            #                 print(f"[DEBUG] First param with grad: {name}, grad_norm: {p.grad.norm().item():.6e}")
            #                 any_present = True
            #                 break
            #         print(f"[DEBUG] Any grad present after backward? {any_present}")
            # ------------------------------------------------------------

           

        
        # Optimizer step
        model_engine.step()

        # --- CHECKSUM AFTER STEP (place immediately after model_engine.step()) ---
        if config.master_process:
            checksum_after = param_checksum(model_engine.module)
            delta = checksum_after - checksum_before
            print(f"[DEBUG] Step {step} checksum_after: {checksum_after:.6e}, delta: {delta:.6e}")
        # ------------------------------------------------------------

       

        # Learning rate scheduling
        if model_engine.lr_scheduler is not None:
            model_engine.lr_scheduler.step()
        
        # Synchronize and calculate timing
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        
        # Token counting
        tokens_processed_local = train_loader.B * train_loader.seq_len * config.accum_steps
        tokens_processed_total = tokens_processed_local * config.world_size
        tokens_per_sec = tokens_processed_total / dt
        total_tokens_seen += tokens_processed_total
        
        # Enhanced logging with expert utilization
        if config.master_process:
            avg_ce_loss = ce_loss_accum / config.accum_steps
            avg_aux_loss = aux_loss_accum / config.accum_steps
            
            # Expert utilization stats
            expert_util_str = ""
            if expert_counts_total is not None:
                expert_counts_avg = expert_counts_total.float() / config.accum_steps
                expert_util_str = f", Expert usage: {expert_counts_avg.tolist()}"
            
            log_params.step = step
            log_params.shard_idx = shard_idx
            log_params.loss_accum = avg_ce_loss.item()
            log_params.lr = model_engine.get_lr()[0]
            
            # Log training loss and learning rate
            eval_log_utils.TrainLoss(log_params=log_params).log_training_loss()
            eval_log_utils.LearningRate(log_params=log_params).log_learning_rate()
            
            # Print progress with expert usage
            print(f"Step {step}, shard_idx: {shard_idx}, CE Loss: {avg_ce_loss.item():.5f}, "
                  f"Aux Loss: {avg_aux_loss.item():.6f}, LR: {model_engine.get_lr()[0]:.7f}, "
                  f"Time: {dt:.2f}sec, Tokens/sec: {tokens_per_sec:,.0f}{expert_util_str}")
        
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


# python train_moe_deepspeed.py
# %%
