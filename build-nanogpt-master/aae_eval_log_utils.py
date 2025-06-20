#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import numpy as np
from hellaswag import render_example, iterate_examples
import csv


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
    def __init__(self, model=None, device='cuda', ddp_world_size=1, ddp_rank=0, log_file='log.txt', hella_loss_file='hella_loss.csv', step=0):
        self.model = model
        self.device = device
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.log_file = log_file

    def get_most_likely_row(tokens, mask, logits):
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

#     def compute_accuracy(self, model, dataloader, device):
#         num_correct_norm = 0
#         num_total = 0
#         for i, example in enumerate(iterate_examples("val")):
#             # only process examples where i % ddp_world_size == ddp_rank
#             if i % ddp_world_size != ddp_rank:
#                 continue
#             # render the example into tokens and labels
#             _, tokens, mask, label = render_example(example)
#             tokens = tokens.to(device)
#             mask = mask.to(device)
#             # get the logits
#             with torch.no_grad():
#                 with torch.autocast(device_type=device, dtype=torch.bfloat16):
#                     logits, loss = model(tokens)
#                 pred_norm = get_most_likely_row(tokens, mask, logits)
#             num_total += 1
#             num_correct_norm += int(pred_norm == label)
#         # reduce the stats across all processes
#         if ddp:
#             num_total = torch.tensor(num_total, dtype=torch.long, device=device)
#             num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
#             dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
#             dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
#             num_total = num_total.item()
#             num_correct_norm = num_correct_norm.item()
        
#         acc_norm = round((num_correct_norm / num_total), 4) if num_total > 0 else 0.0
        
#         if master_process:
#             print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
#             with open(log_file, "a") as f:
#                 f.write(f"{step} hella {acc_norm:.4f}\n")

#             with open(hella_loss_file, "a") as f:
#                 csv_out = csv.writer(f)
#                 csv_out.writerow([step, acc_norm])

#%%
   

class validation_check:
    def __init__(self, model=None, val_loader=None, ddp=True, rank=None):
        self.model = model
        self.val_loader = val_loader
        self.ddp = ddp
        self.rank = rank


        pass