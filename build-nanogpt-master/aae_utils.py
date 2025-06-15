# %%
# class to create data loader
import torch
import torch.nn as nn
import tiktoken
import os
import numpy as np

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load the text file for training. Encode the text and convert it to a tensor
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'Batch size: {B}, Sequence length: {T}')
        print(f'Tokens per batch: {(self.B * self.T)}')
        self.batches_per_epoch = len(self.tokens) // (self.B * self.T)

        # this keeps track of where we are in the text for batching
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        self.current_position += B*T # update the current position in the text
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(B, T) # input sequence
        y = buf[1:].view(B, T) # target sequence

        # if loading the next batch would go beyond the end of the training text, reset to the beginning of the text. IT SEEMS TO ME THAT THIS APPROACH LEAVES OUT SOME OF THE LAST TOKENS IN THE TEXT. SO WHATEVER PORTION OF THE TEXT IS NOT USED IN THE LAST BATCH, IT IS NOT USED AT ALL BEACUSE WE RESET TO THE BEGINNING OF THE TEXT.
        if self.current_position + B*T + 1 > len(self.tokens):
            # reset to the beginning of the text
            self.current_position = 0
        
        return x, y
    
class DataLoaderMultiGPU:
    def __init__(self, B, T, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'Batch size: {B}, Sequence length: {T}')
        print(f'Tokens per batch: {(self.B * self.T)}')

        # this keeps track of where we are in the text for batching. Note that the current position is multiplied by the process RANK to ensure that each process gets a different part of the text. The text assigned to each process remains the same throughout training. 
        self.current_position = self.B * self.T * self.process_rank 

    def next_batch(self):
        B, T = self.B, self.T

        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(B, T) # input sequence
        y = buf[1:].view(B, T) # target sequence

        self.current_position += B * T * self.num_processes # update the current position in the text

        # if loading the next batch would go beyond the end of the training text, reset to the beginning of the text
        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            # reset to the beginning of the text that was assisgned to this process.
            self.current_position = self.B * self.T * self.process_rank 
        
        
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        return x, y
    

#%%
#create dataloader for processing shards on multi gpu
class DataLoaderShardMultiGPU:
    def __init__(self, B, T , process_rank=0, num_processes=1, split=None, shard_dir='edu_fineweb10B'):
        self.B = B
        self.T = T
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.shard_dir = shard_dir
        self.remaining_tokens = []  # to capture the tokens at the end of the current shard that will not be used or seen
        assert split in {'train', 'val'}, f'you must specify if the data is train or val'

        # returns an unordered list of file names in the shard directory. Note that these are just string file names, not the actual numpy arrays.
        self.shard_files = os.listdir(shard_dir) 

        # filter the shard file names to only include those that match the split and sort
        self.shard_files = [file for file in self.shard_files if split in file]
        self.shard_files.sort() 
        

        # this is Karpathy's code. I don't understand why he does this. we already have the shard files in a sorted list and can just add the shard_dir to the file names to get the full path when loading as I have done in the load_tokens_convert_to_tensor method.
        # self.shards = [os.path.join(shard_dir, file) for file in self.shard_files]
        
        self.reset()  # initialize the current shard index and load the first shard

    
    # Karpathy puts this function outside the class, I think it cleaner to have it as a method of the class. This function loads the tokens from a shard file and converts them to a tensor. 
    def load_tokens_convert_to_tensor(self, shard_file):
        shard_numpy = np.load(f'{self.shard_dir}/{shard_file}')

        # Not sure why karpathy does this. code below recasts to torch.long which is equivalent to int64 in numpy. if his intention was to have int32 tensor, this does not do that. So I've commented it out.
        # shard_numpy = shard_numpy.astype(np.int32)  

        shard_tensor = torch.tensor(shard_numpy, dtype=torch.long)  # convert to tensor with dtype int32
        return shard_tensor
        
    def reset(self):
        self.current_shard_idx = 0
        self.shard_tensor = self.load_tokens_convert_to_tensor(self.shard_files[self.current_shard_idx])
        self.current_position = self.B * self.T * self.process_rank  # reset the current position in the text for this process
        
    def next_batch(self):
        B, T = self.B, self.T

        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.shard_tensor[self.current_position:self.current_position + B*T + 1]
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(B, T) # input sequence
        y = buf[1:].view(B, T) # target sequence

        self.current_position += B * T * self.num_processes # update the current position in the text   

        # if loading the next batch would go beyond the end of the current shard, move to the next shard
        if self.current_position + (B*T*self.num_processes + 1) > len(self.shard_tensor):
            # really clever code from Karpathy. If currently at the last shard, it produces the index of the next shard file in the list. If currently at last file in the shard file list, it sets the index for the next shard to 0. Very clever  way to loop back to the first file after reaching the end of the shard files list.
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)

            # capture the tokens at the end of the current shard that will not be used or seen. Note that in Karpathy's implementation, these tokens are just never used.
            remaining_tokens = self.shard_tensor[self.current_position:]
            self.remaining_tokens.append(len(remaining_tokens))
            
            # load the next shard
            self.shard_tensor =  self.load_tokens_convert_to_tensor(self.shard_files[self.current_shard_idx])  
            self.current_position = self.B * self.T * self.process_rank  # reset the current position in the text for this process
        
        
        # move the tensors to the GPU
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        
        return x, y, self.current_shard_idx, self.remaining_tokens
   

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