# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from hellaswag import render_example, iterate_examples


   

# %%
# class to configure the optimizer

# NOTE: In PyTorch's AdamW optimizer, param_group is a dictionary specifying the tensors to be optimized along with group-specific optimization options. It allows for applying different hyperparameters, such as learning rate or weight decay, to different sets of parameters within the model. This is useful for fine-tuning specific layers or parts of the network with varying optimization strategies. optimizer.param_groups is a list of such dictionaries, where each dictionary corresponds to a different group of parameters. The optimizer iterates over these groups during the optimization process, applying the specified settings to each group.

# configure_optimizers is a method that creates parameter groups (param_groups) for the optimizer  to apply weight decay only to tensors that are involved in matmul (excludes bias and layernorm tensors). This is a performance optimization that allows for more efficient training on GPUs. It creates two parameter groups: one for the weights and one for the biases. The weights are optimized with weight decay and the biases and layernorms are optimized without weight decay. This is done to improve generalization and reduce overfitting. The method also sets the learning rate and weight decay for each parameter group.

# Note that my implementation of this method is different from Karpathy's.

class ConfigureOptimizer:
    def __init__(self, model):
        self.model = model

    def create_optimizer(self, weight_decay=0.01, learning_rate=6e-4, device_type= None):

        assert device_type is not None, 'a device_type must be specified'
        # create parameter groups for the optimizer to apply weight decay only to tensors that are not bias or layernorm.

        decay_params = set()
        no_decay_params = set()
        for name, parameter in self.model.named_parameters():  
            if 'bias' in name or 'ln' in name:
                no_decay_params.add(name)
            else:
                decay_params.add(name)
        param_groups = [
            {"params": [parameter for name, parameter in self.model.named_parameters() if name in decay_params], "weight_decay": weight_decay},
            {"params": [parameter for name, parameter in self.model.named_parameters() if name in no_decay_params], "weight_decay": 0.0}
        ]
        
        # use fusion for the optimizer if the device is cuda. This is a performance optimization that allows for more efficient training on GPUs. when fusion is available, PyTorch internally uses torch._foreach APIs or custom fused CUDA kernels. It avoids multiple reads/writes to GPU memory per parameter per step. Can lead to significant speedups in large-scale training tasks (e.g., transformer models with billions of parameters).
        if device_type == 'cuda':
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay, eps=1e-8, fused=True)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay, eps=1e-8, fused=False)
        
        return optimizer

# %%
# create learning rate scheduler class
# This my implementation of the cosine learning rate scheduler and is different from Karpathy's implementation. I am using the Pytorch implementatons of cosine annealing scheduler with and without out restart as well as my own code for an initial linear warm-up. the user has to option use cosine annealing with restarts or not, as well as the option to use a linear warmup or not with the warmup steps as a parameter

class CosineLearingRateScheduler:
    def __init__(self, 
                 optimizer = None,
                 T_max = 50, restart = False, warm_up_steps =10,  max_lr=6e-4, 
                 min_lr = 1e-5, T_mult=1, T_0 = 10):
        
        self.optimizer = optimizer
        self.T_max = T_max
        self.warm_up_steps = warm_up_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.restart = restart
        self.T_mult = T_mult
        self.T_0 = T_0

        assert self.optimizer is not None, 'an optimizer object must be provided'

        self.lrs =[]
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        if self.restart:
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)
        else:
            self.scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, 
                                               eta_min=self.min_lr)
     
    def set_lr(self, step):
        if step < self.warm_up_steps:
            # Linear warmup: scale up from 0 to max_lr
            warmup_lr = self.max_lr * (step + 1) /self.warm_up_steps
            # print(f'setting warmup lr to {warmup_lr}')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        if step >= self.warm_up_steps:
            # Step the cosine scheduler
            self.scheduler.step(step - self.warm_up_steps)
        
        self.lrs.append(self.optimizer.param_groups[0]['lr'])

class DDP_Setup(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Check if we are running in DDP mode. If so, we will initialize the process group and set the device for each process.

        # a simple way to check whether your script is being run under Distributed Data Parallel (DDP) — specifically when using torchrun with a cuda GPU. Note that you can be in DDP mode even with a single GPU when using torchrun. 
        self.ddp_available = int(os.environ.get('RANK', -1)) != -1
        self.setup()
    
    def setup(self):
        if self.ddp_available:
            print(f'\nRunning in Distributed Data Parallel (DDP) mode')
            # Note that LOCAL_RANK is the rank of the process on one given machine (when using multiple machine), while RANK is the rank of the process across all machines (when using multiple gpus on multiple machines). When using a setup with just one machine, LOCAL_RANK and RANK are the same. 
            init_process_group(backend='nccl') # initialize the process group for DDP
            self.ddp_rank = int(os.environ['RANK']) # get the rank of the current process
            self.ddp_local_rank = int(os.environ['LOCAL_RANK']) # get the local rank of the current process
            self.ddp_world_size = int(os.environ['WORLD_SIZE']) # get the total number of processes
            
            # set the device to the local rank of the current process
            self.device = f'cuda:{self.ddp_local_rank}' 
            torch.cuda.set_device(self.device) # set the device for the current process

            # the master process will perform logging and saving checkpoints.
            self.master_process = (self.ddp_rank == 0)

        # if not using DDP, just use the next best availabel option
        else: 
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.ddp_rank = 0
                self.ddp_local_rank = 0
                self.ddp_world_size = 1
                self.master_process = True

            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
                
            print(f"\nusing device: {self.device}")

        torch.manual_seed(42) # set the random seed for reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42) # set the random seed for cuda for reproducibility
    
    def wrap_model(self):
        DDP(self.model, device_ids=[self.ddp_local_rank], find_unused_parameters=False)


    # %%
    #Instantiate the model and implement torch.compile if cuda is available.

    # if cuda is available, use torch.compile to optimize the model for training on GPUs. This is a performance optimization that allows for more efficient training on GPUs. It uses the PyTorch JIT compiler to optimize the model for the specific hardware and software configuration. This is done to improve performance and reduce memory usage. we use bfloat16 precision for the forward pass and use torch.compile. See Karpathy's tutorial at 1:24:00 and 1:49:00 for details


    

# class RotaryPosEmbed:
#     def __init__(self, seq_len=1024, head_dim=768):
#         self.seq_len = seq_len
#         self.head_dim = head_dim
#         # self.device = device

#     # Compute rotary angles
#     def get_angels(self, device = 'cuda'):
#         theta = 10_000 ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) / self.head_dim)
#         pos = torch.arange(self.seq_len, dtype=torch.float, device=device)
#         angles = torch.outer(pos, theta)
#         return angles
    
#     # x is the input vector with shape: [batch_size, seq_length, num_heads, head_dim]
#     def apply_rotation(self, x=None):
#         device = f'cuda:{x.get_device()}'
#         # print(f'device: {device}')
#         angles = self.get_angels()

#         # Apply sin and cos to angles and use unsqueeze to add dimensions to match number of dimensions of input vector 
#         sin = angles.sin().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
#         cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]

#         # split input vector x into two vectors from the  even and odd indexed elements of the original vector x. Each element from x1 and x2 will be paired for rotation
#         x1 = x[:, :, :, : :2]
#         x2 = x[:, :, :, 1: :2]

#         # Apply rotation. Note that the elementwise multiplication broadcasts the sin and cos values into batch and num_heads dimensions
#         x1_rot = x1 * cos - x2 * sin #[B, S, num_heads,  head_dim/2]
#         x2_rot = x1 * sin + x2 * cos #[B, S, num_heads,  head_dim/2]

#         # Stack into [B, S, head_num, head_dim/2, 2] the dim=-1 adds a new dimension to the end of [B, S, H, head_dim/2] and stacks each corresponding element from dim=1 of [seq_length, dim/2] from the x_rotated_even and x_rotated_odd vectors into that new third dimension
#         x_rot = torch.stack([x1_rot, x2_rot], dim=-1) #[B, S, H, head_dim/2, 2]

#         # flatten last two dims back to [B, seq_len, num_head, head_dim]
#         x_rot = x_rot.flatten(start_dim=3, end_dim=-1) 
        
#         return x_rot



# %%
