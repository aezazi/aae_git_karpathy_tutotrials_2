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


class CreateLogFiles:
    def __init__(self, loss_dir="train_loss", hella_accu_dir="hella_accuracy", learn_rate_dir = 'learn_rate_sched'):
        self.loss_dir = loss_dir
        self.hella_accur_dir = hella_accu_dir
        self.learn_rate_dir = learn_rate_dir

    
        # create traing loss log directory
        os.makedirs(self.loss_dir, exist_ok=True)
        self.train_loss_file = os.path.join(self.loss_dir, f"train_loss.csv")
        with open(self.train_loss_file, "w") as f: # open for writing to clear the file
            csv_out = csv.writer(f)
            csv_out.writerow(['step', 'train_loss']) # write the header row

        # create traing loss log directory
        os.makedirs(self.hella_accur_dir, exist_ok=True)
        self.hella_accu_file = os.path.join(self.hella_accur_dir, f"hellaswag_eval.csv")
        with open(self.hella_accu_file, "w") as f: # open for writing to clear the file
            csv_out = csv.writer(f)
            csv_out.writerow(['step', 'hellaswag_accuracy']) # write the header row
    
    def log_training_loss(self, step=None, loss_accum=0):
        with open(self.train_loss_file, "a") as f:
                csv_out = csv.writer(f)
                csv_out.writerow([step, f'{loss_accum.item():.7f}']) # write the step and loss to the csv file

        
class HellaSwag:
    def __init__(self, model=None, device='cuda', ddp_world_size=1, ddp_rank=0, hella_accu_file='hella_eval.csv', step=0, ):
        self.model = model
        self.device = device
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.log_hella_accufile = hella_accu_file
        self.master_process = ddp_rank == 0

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

    def compute_accuracy(self, ddp=True):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % self.ddp_world_size != self.ddp_rank:
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
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            self.num_total = num_total.item()
            self.num_correct_norm = num_correct_norm.item()
        
        self.acc_norm = round((self.num_correct_norm / self.num_total), 4) if self.num_total > 0 else 0.0
        
    def log_hella_accu(self, step=1, log_file=None):
        self.compute_accuracy(ddp=True)
        if self.master_process:
            print(f"HellaSwag accuracy: {self.num_correct_norm}/{self.num_total}={self.acc_norm:.4f}")
        
            with open(log_file, "a") as f:
                csv_out = csv.writer(f)
                csv_out.writerow([step, self.acc_norm])

#%%
   

class validation_check:
    def __init__(self, model=None, device = "cuda", optimizer=None, val_loader=None, ddp=True, ddp_rank=1, step=0):
        self.model = model
        self.val_loader = val_loader
        self.ddp = ddp
        self.rank = ddp_rank
        self.device = device
        self.optimizer = optimizer
        self.master_process = ddp_rank == 0

    def check_validation_loss(self):
        self.model.eval() # set the model to evaluation mode
        self.val_loader.reset() # reset the validation loader to the beginning of the validation dataset
        
        with torch.no_grad(): # no need to compute gradients for validation
            val_loss_accum = 0.0
            val_loss_steps = 20 # number of steps to accumulate validation loss over
            for _ in range(val_loss_steps):
                x, y, shard_idx, tokens_abandoned = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)

                # see training loop below for details on the use of autocast. 
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, val_loss = self.model(x, y)
                
                val_loss = val_loss / val_loss_steps # divide the loss by the number of accumulation steps to get the average loss. This computes the averaage loss on one gpu.
                val_loss_accum += val_loss.detach() # detach the loss from the computation graph to avoid memory leaks.

        if self.ddp:
            # synchronize the validation loss across all gpu processes
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        
        if self.master_process:
            print(f"Validation at Step {self.step},  shard_idx: {shard_idx},  Loss: {val_loss_accum.item():.4f},  LR: {self.self.optimizer.param_groups[0]['lr']:.7f}")