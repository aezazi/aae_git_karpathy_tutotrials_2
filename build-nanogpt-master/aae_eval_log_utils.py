#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import numpy as np
from hellaswag import render_example, iterate_examples
import csv
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataclasses import dataclass

@dataclass
class LogParamsFilesConfig:
    fsdp_ddp: bool 
    world_size: int 
    rank: int 
    local_rank: int
    model: object 
    device: str
    encoder: object
    # optimizer: object 
    val_loader: object 
    loss_dir: str
    hella_accu_dir: str
    learn_rate_dir: str
    train_loss_file: str
    hella_accu_file: str 
    lr_file: str
    step: int
    shard_idx: int
    loss_accum: float 
    lr: float

    def __post_init__(self):
        # create traing loss log directory
        os.makedirs(self.loss_dir, exist_ok=True)
        self.train_loss_file = os.path.join(self.loss_dir, self.train_loss_file)
        with open(self.train_loss_file, "w") as f: # open for writing to clear the file
            csv_out = csv.writer(f)
            csv_out.writerow(['step', 'train_loss']) # write the header row

        # create hellaswag accuracy log directory
        os.makedirs(self.hella_accu_dir, exist_ok=True)
        self.hella_accu_file = os.path.join(self.hella_accu_dir, self.hella_accu_file)
        with open(self.hella_accu_file, "w") as f: # open for writing to clear the file
            csv_out = csv.writer(f)
            csv_out.writerow(['step', 'hellaswag_accuracy']) # write the header row

        # create learning rate log directory
        os.makedirs(self.learn_rate_dir, exist_ok=True)
        self.lr_file = os.path.join(self.learn_rate_dir, self.lr_file)
        with open(self.lr_file, "w") as f: # open for writing to clear the file
            csv_out = csv.writer(f)
            csv_out.writerow(['step', 'learning_rate']) # write the header row
 

class TrainLoss():
    def __init__(self, log_params):
        self.step = log_params.step
        self.loss_accum = log_params.loss_accum
        self.train_loss_file = log_params.train_loss_file
        self.master_process = log_params.rank == 0
    
    def log_training_loss(self):
        with open(self.train_loss_file, "a") as f:
                csv_out = csv.writer(f)
                csv_out.writerow([self.step, self.loss_accum]) # write the step and loss to the csv file

class LearningRate():
    def __init__(self, log_params):
        self.step = log_params.step
        self.lr = log_params.lr
        self.lr_file = log_params.lr_file
        self.master_process = log_params.rank == 0

    def log_learning_rate(self):
        with open(self.lr_file, "a") as f:
                csv_out = csv.writer(f)
                csv_out.writerow([self.step, f'{self.lr:.7f}'])

class HellaSwag:
    def __init__(self, log_params):
        self.model = log_params.model
        self.device = log_params.device
        self.fsdp_ddp = log_params.fsdp_ddp
        self.world_size = log_params.world_size
        self.rank = log_params.rank
        self.master_process = log_params.rank == 0
        self.step = log_params.step
        self.log_file = log_params.hella_accu_file
        
    def get_most_likely_row(self,tokens=None, mask=None, logits=None):
    # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred_norm = avg_loss.argmin().item()
        return pred_norm  

    def compute_accuracy(self):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % world_size == ddp_rank
            if i % self.world_size != self.rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    logits, loss = self.model(tokens)
                pred_norm = self.get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if self.fsdp_ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            self.num_total = num_total.item()
            self.num_correct_norm = num_correct_norm.item()
        
        self.acc_norm = round((self.num_correct_norm / self.num_total), 4) if self.num_total > 0 else 0.0
        
    def log_print_hella_accuracy(self):
        self.compute_accuracy()
        if self.master_process:
            print(f"\nHellaSwag accuracy at Step {self.step}: {self.num_correct_norm}/{self.num_total}={self.acc_norm:.4f}\n")
        
            with open(self.log_file, "a") as f:
                csv_out = csv.writer(f)
                csv_out.writerow([self.step, self.acc_norm])

#%%
class Validation:
    def __init__(self, log_params=None):
        self.model = log_params.model
        self.val_loader = log_params.val_loader
        self.fsdp_ddp = log_params.fsdp_ddp
        self.device = log_params.device
        self.master_process = log_params.rank == 0
        self.step = log_params.step
        self.shard_idx = log_params.shard_idx
        self.lr = log_params.lr

    def check_validation_loss(self):
        self.model.eval() # set the model to evaluation mode
        self.val_loader.reset() # reset the validation loader to the beginning of the validation dataset
        # self.step = step
        
        with torch.no_grad(): # no need to compute gradients for validation
            val_loss_accum = 0.0
            val_loss_steps = 20 # number of steps to accumulate validation loss over
            for _ in range(val_loss_steps):
                x, y, shard_idx, tokens_abandoned = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)

                # see training loop for details on the use of autocast. 
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, val_loss = self.model(x, y)
                
                val_loss = val_loss / val_loss_steps # divide the loss by the number of accumulation steps to get the average loss. This computes the averaage loss on one gpu.
                val_loss_accum += val_loss.detach() # detach the loss from the computation graph to avoid memory leaks.

        if self.fsdp_ddp:
            # synchronize the validation loss across all gpu processes
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if self.master_process:
                print(f"\nValidation loss  at Step {self.step},  shard_idx: {self.shard_idx},  Loss: {val_loss_accum:.4f},  LR: {self.lr:.7f}\n")
         
class GenerateSample:
    def __init__(self, log_params):
        self.model = log_params.model
        self.device = log_params.device
        self.rank = log_params.rank
        self.enc = log_params.encoder

    def generate(self, context="Hello, I'm a language model,", sample_max_length=32):  
        self.model.eval()
        self.num_return_sequences = 4
        # self.max_length = sample_max_length
        self.tokens = self.enc.encode(context)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.tokens = self.tokens.unsqueeze(0).repeat(self.num_return_sequences, 1)
        self.xgen = self.tokens.to(self.device)
        self.sample_rng = torch.Generator(device=self.device)
        self.sample_rng.manual_seed(42 + self.rank)

        while self.xgen.size(1) < sample_max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    logits, loss = self.model(self.xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=self.sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                self.xgen = torch.cat((self.xgen, xcol), dim=1)
        # print the generated text
        for i in range(self.num_return_sequences):
            self.tokens = self.xgen[i, :sample_max_length].tolist()
            decoded = self.enc.decode(self.tokens)
            print(f"\nrank {self.rank} sample {i}: {decoded}\n")
