#%%
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import aae_utils

#%%
# # experiment with top-k gating for MoE layers
# # This is a simple example of how to implement top-k gating for MoE layers in PyTorch. The top-k gating mechanism selects the top-k experts based on the logits computed from the multi-head attention output.
# num_experts = 5
# top_k=2
# n_embed=8
# seq_len = 7
# batch_size = 3

# torch.random.manual_seed(42)  # for reproducibility

# # Simulate multi-head attention output. In practice, this would be the output from a multi-head attention layer.
# mh_output = torch.randn(batch_size, seq_len, n_embed)

# # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each expert. the logits will have shape (batch_size, seq_len, num_experts) 
# topkgate_linear = nn.Linear(n_embed, num_experts) 

# # In each batch, there is a logit for each token in the sequence and for each expert. 
# logits = topkgate_linear(mh_output) #(batch_size, seq_len, num_experts) 
# print(f'logits: \n{logits}\n')  

# # Get the top-k logits and their corresponding indices. The top_k function returns the top-k values and their indices along the last dimension (num_experts). In each batch, for each token in the sequence, it selects the top-k logits and their indices from the logits produced by each expert.
# # The logits  and their indices will have shape (batch_size, seq_len, top_k) 
# top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)  
# print(f'top_k_logits with shape {top_k_logits.shape}: \n{top_k_logits}\n') # (batch_size, seq_len, top_k)
# print(f'top_k_indices with shape {top_k_indices.shape}: \n{top_k_indices}\n')

# # We want sparse matrices. We achieve this by keeping the top-k logits for each token and filling the rest with a value that represents "no contribution" (like negative infinity).  This is done to ensure that the softmax function will ignore these values when computing the probabilities for each expert. So only the values produced by the top-k experts will contribute to the final output. We implement this by first creating a tensor of the same shape as the logits filled with negative infinity, and then using the scatter function to fill in the top-k logits at the indices of the top-k experts.
# zeros = torch.full_like(logits, float('-inf')) #full_like clones a tensor and fills it with a specified value (like infinity) for masking or calculations.
# sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
# print(f'\nsparse_logits:\n {sparse_logits}\n')

#%%
@dataclass
class GPTConfig:
    seq_len: int = 1024 # max sequence length
    # setting vocab size to 50304 rather than 50257 (the size of the gpt2 vocab) because this is a much more efficient number (divisible by many powers of 2) for gpu kernels and computations. The extra tokens are just padding tokens that are not used in the model. The model will learn to ignore them. this is a tradeoff between memory and performance. 
    seq_len: int = 7
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 8
    num_experts = 5
    k = 2

# instantiate and check the config
config = GPTConfig()

#%%
#This class creates a single expert in the MoE layer. Each expert is a simple feedforward network with a swiglu activation function.
class ExpertMoESwiglu(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.n_embd * 4 # hidden dimension for the expert
        self.ln_1 = nn.Linear(config.n_embd, self.hidden_dim) 
        self.ln_1._am_expert =True
        self.ln_2 = nn.Linear(config.n_embd, self.hidden_dim)
        self.ln_2._am_expert =True
        self.c_proj = nn.Linear(self.hidden_dim, config.n_embd)
        self.c_proj._am_expert =True
        # torch.manual_seed(42)

    def forward(self, x):
        x =self.ln_1(x) * F.silu(self.ln_2(x))  # this is Swiglu activation
        x= self.c_proj(x)
        return x

# this class implemets top_k sparse gating 
class TopKMoEGate(nn.Module):
    def __init__(self, config, local_expert_ids):
        super().__init__()
        self.local_expert_ids = local_expert_ids
        self.num_local_experts = len(self.local_expert_ids)
        self.k = config.k
        self.seq_len = config.seq_len
    
        # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each local expert. The local experts are the experts assigned to the GPU running this instance. This is very important. I had  orignally made the mistake of doing nn.Linear(config.n_embd, self.num_experts, bias=False) the logits will have shape (batch_size, seq_len, num_experts) for all experts. This was computing logits over all num_experts, regardless of how many experts exist on each GPU.But the MoE layer (MoELayerSharded) only instantiates a subset of those experts per GPU (based on expert_id % world_size == local_rank), and applies only the ones available on that rank.

        # This results in:
        # 	• These tokens are ignored during the expert loop, i.e., dropped silently.
        # 	• Tokens being routed to experts that don’t exist locally.
        # 	• Thus, part of the model output remains zeros.
        # 	• Consequently, gradients don’t flow to many experts.
        # 	• Eventually, the gating network settles into a degenerate distribution over the few working experts.
        # 	• This is why my  loss was stalling at ~8.2 on multi-GPU — it’s only marginally better than random output.This doesn’t happen on single-GPU because all experts exist and are used, so the full distribution of the gating network is valid.
        self.gate_linear = nn.Linear(config.n_embd, self.num_local_experts, bias=False)

        # this is a learnable parameter that will be used to scale the noise added to the logits of each expert. The allows the noise added to each expert to be "customized" and dynamic. It adapts during training depending on the expert. It is initialized to zero and will be learned during training.
        self.noise_weight = nn.Parameter(torch.zeros(self.num_local_experts)) 

        # this is the standard deviation of the noise added to the logits of each expert. It is a hyperparameter that can be tuned. Note that unlike the noise_weight it is a global parameter, this is not a learnable parameter.
        self.noisy_std = 1.0
        
    # x has shape (batch_size, sequence_length, embedding dimension) and is the output of the multi-head attention layer. 
    def forward(self, x):
        
        # In each batch, there is a logit for each token in the sequence and for each expert. 
        logits = self.gate_linear(x) # (batch_size, seq_len, num_local_experts)
        print(f'\nshape logits: {logits.shape}:\n{logits}\n')

        # The global noise to be added to the logits of each expert. This is a random noise tensor of the same shape as the logits. 
        noise = torch.randn_like(logits) * self.noisy_std # (batch_size, seq_len, num_experts)

        # Per-expert noise scaling using self.weights. 
        noise = noise * self.noise_weight

        # Add the noise to the logits. 
        logits_noisy = logits + noise  # (batch_size, seq_len, num_experts)

        # Get the top-k logits and their corresponding indices. The pytorch top_k method returns the top-k values and their indices along the last dimension (num_experts). In each batch, for each token in the sequence, it selects the top-k logits and their indices from the logits produced by each expert.
        top_k_logits_noisy, top_k_indices_noisy = logits_noisy.topk(self.k, dim=-1)  # (batch_size, seq_len, top_k) 

        # We want sparse matrices. We achieve this by keeping the top-k logits for each token and filling the rest with a value that represents "no contribution" (like negative infinity).  This is done to ensure that the softmax function will ignore these values when computing the weights for each expert. So only the values produced by the top-k experts will contribute to the final output. We implement this by first creating a tensor of the same shape as the logits filled with negative infinity, and then using the scatter function to fill in the top-k logits at the indices of the top-k experts. Note that in this context, softmax is being used to compute "weights" for each expert not probabilities as for multiclass classification. Its a subttle difference but I think important to note.
        
        #full_like clones a tensor and fills it with a specified value (like infinity).
        mask = torch.full_like(logits_noisy, float('-inf')) 

        # scatter fills the zeros tensor with the top_k_logits at the indices of the top_k_indices along the last dimension (-1). This creates a sparse matrix where only the top-k logits are kept and the rest are filled with negative infinity.
        sparse_logits_noisy = mask.scatter(-1, top_k_indices_noisy, top_k_logits_noisy)
        top_k_gated_weights = F.softmax(sparse_logits_noisy, dim=-1)

        return top_k_gated_weights, top_k_indices_noisy


# %%
# test the classes above
batch_size = 3
example_input = torch.randn(batch_size, config.seq_len, config.n_embd)  # Simulated multi-head attention output
lcl_expert_ids = torch.tensor([0,1,2,3,4]) 
top_k_gate = TopKMoEGate(config, lcl_expert_ids)
gated_weights, top_k_lcl_indices= top_k_gate(example_input)

print(f'gated_weights shape: {gated_weights.shape} \n{gated_weights}\n') 
print(f'gated_weights_flattened shape: {gated_weights.view(batch_size*config.seq_len, len(lcl_expert_ids)).shape} \n{gated_weights.view(batch_size*config.seq_len, len(lcl_expert_ids))}')


        

# %%
# this class implements the full MoE layer. It uses the TopKMoEGate class to compute the gated probabilities and the top-k indices, and then applies each expert to the input based on these indices. The output is a weighted sum of the expert outputs, where the weights are the gated probabilities.
# class MoELayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_experts = config.num_experts
#         self.sq_len = config.seq_len
#         self.k = config.k
    
#         # Create a linear layer to project the multi-head attention output to the number of experts. This layer will compute the logits for each expert. the logits will have shape (batch_size, seq_len, num_experts) 
#         self.gate = TopKMoEGate(config)

#         # Create a list of experts, each expert is an instance of the ExpertMoESwiglu class
#         self.experts = nn.ModuleList([ExpertMoESwiglu(config) for _ in range(self.num_experts)])
       
#         torch.manual_seed(42)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         # Get the gated probabilities and top-k indices from the gate
#         gated_weights, top_k_indices, top_k_logits = self.gate(x)

#         # Initialize the fianl output tensor
#         final_output = torch.zeros_like(x)
#         print(f'\ninput x shape: {x.shape} \n{x}\n')

#         # flatten the input to (batch_size * seq_len, n_embd) for batch processing
#         x_flat = x.view(batch_size*seq_len, -1) 
#         print(f'\ninput x_flat shape: {x_flat.shape} \n{x_flat}\n')

#         # flatten the gated probabilities to (batch_size * seq_len, num_experts) for batch processing
#         gated_weights_flat = gated_weights.view(batch_size*seq_len, self.num_experts)  

        

#         # Iterate over each expert and apply it to the input
#         for i, expert in enumerate(self.experts):
#             # Create a mask for the inputs where the current expert is in top-k. the mask will have shape (batch_size, seq_len). top_k_indices have shape (B, seq_len, top_k). Each row of each batch in top_k_indices has the indices of the top two experts for the token corresponding that row in the token sequence. The mask will return True (row wise) if expert i is in the top_k indices 
#             print(f'\nExpert {i} with x_flat input shape {x_flat.shape}\n')
#             print(f'top_k_indices shape: {top_k_indices.shape} \n{top_k_indices}\n')
#             expert_mask = (top_k_indices == i).any(dim=-1)
#             print(f'expert_mask shape: {expert_mask.shape} \n{expert_mask}\n')

#             # flatten the mask to match the shape of the flattened input x_flat. Note that the shape of flat_mask is a one dimensional (batch_size*seq_len). x_flat has shape (batch_size * seq_len, n_embd). each row in x_flat is a token in the sequence. 
#             flat_mask = expert_mask.view(-1) # (batch_size * seq_len)
#             print(f'flat_mask shape: {flat_mask.shape} \n{flat_mask}\n')

#             if flat_mask.any():
#                 # Apply the expert to the inputs selected by the mask. x_flat[flat_mask] picks the rows(tokens) of x_flat where the mask is True. This allows us to activate the expert only for the tokens where the expert is in the top_k indices.
#                 expert_input = x_flat[flat_mask] # (number of tokens where expert i is in top_k, n_embd)
#                 print(f'\nexpert_input shape: {expert_input.shape} \n{expert_input}\n')
                
#                 # apply expert i to the expert_input. Again, note that based on the mask described above, epxert i is applied only to the tokens where it is in the top_k indices.
#                 expert_output = expert(expert_input) # (number of tokens where expert i is in top_k, n_embd)
#                 print(f'expert_output shape: {expert_output.shape} \n{expert_output}\n')

#                 # Now we need to scale the expert output by the gated weights for the tokens where the expert is in the top_k indices. gated_weights_flat has shape (batch_size * seq_len, num_experts). We apply the same mask as we used to create expert_input to select all rows from gated_weights_flat where expert i is in the top_k indices, then we select the ith column. This returns the weighting for expert i that is to be applied to the tokens where expert i is in the top_k indices. This is the same as selecting all the non_zero values in the ith column of gated_weights_flat. We  then use unsqueeze(1) to add a dimension to create a column vector of shape (number of tokens where expert i is in top_k, 1). this allows  multiplication with the expert_output which has shape (number of tokens where expert i is in top_k, n_embd). 
#                 print(f'gated_weights_flat shape: {gated_weights_flat.shape} \n{gated_weights_flat}\n')
#                 expert_weights = gated_weights_flat[flat_mask, i].unsqueeze(1)  # (number of tokens where expert i is in top_k, 1)
#                 print(f'expert_weights shape: {expert_weights.shape} \n{expert_weights}\n')

#                 # Scale the expert_output by expert_weights.
#                 expert_output_weighted = expert_output * expert_weights # (number of tokens where expert i is in top_k, n_embd)
#                 print(f'expert_output_weighted shape: {expert_output_weighted.shape} \n{expert_output_weighted}\n')

#                 # Now we need to add the expert_output_weighted to final_output at positions where the expert is in the top_k indices. We use expert_mask to select the rows where expert i is in the top_k indices. Note that here we use expert_mask (not flat_mask) with shape (batch_size, seq_len, hidden_dim) to match the shape of final_output. final_output will have shape (batch_size, seq_len, n_embd), the same as input x.
#                 print(f'final_output shape before adding expert_output_weighted:{final_output.shape} \n{final_output}\n')

#                 # the huggingface implementation uses .squeeze(1) to remove any singleton dimensions from  the expert_output_weighted tensor. Not sure why this is needed. I tried removing it and the shapes were still compatible and the result the same
#                 final_output[expert_mask] += expert_output_weighted.squeeze(1) # (batch_size, seq_len, n_embd)
#                 print(f'final_output shape after adding expert_output_weighted:{final_output.shape} \n{final_output}\n')

#                 ## just a test to see if .squeeze(1) is necessary
#                 # final_test = torch.zeros_like(x)
#                 # final_test[expert_mask] += expert_output_weighted
#                 # print(f'{torch.allclose(final_output, final_test)}')

#             break
        
#         # compute auxiliary load balancing loss
#         # first compute the average weight given to each expert by the top_k router. This is how much the router wanted to send to each expert.
#         avg_weight_per_expert = gated_weights_flat.mean(0) # shape(num_experts)
#         print(f'\navg_weight_per_expert:\n {avg_weight_per_expert}\n')

#         # compute average number of tokens processed by each expert. This is  how many tokens were actually sent to each expert. To compute this, I use the gated_weights_flat (batch_size*seq_len, num_experts) with non_zero elements only for tokens where the expert was in the top_k. I replace the non-zero element with 1.0 which serves as a counter for the expert having processed that token. Finally, I take the mean to compute the average number of tokens processed by each expert across all batches.
       
#         # Replaces elements != 0 with 1.0 and takes the mean over (batch*seq_len).result has shape(num_experts)
#         avg_tokens_per_expert = torch.where(gated_weights_flat != 0, 1.0, 0).mean(0)

#         print(f'\navg_tokens_per_expert: \n {avg_tokens_per_expert}\n')


#         # refer to literature on why this formula for the load balancing loss
#         aux_loss = (avg_tokens_per_expert * avg_weight_per_expert).sum() * self.num_experts

#         return final_output, aux_loss
    
# # %%
# # test the MoE class
# batch_size = 3
# example_input = torch.randn(batch_size, config.seq_len, config.n_embd)
# moe_layer = MoELayer(config)


# # output, aux_loss = moe_layer(example_input)

# # print(moe_layer.__getattr__)
# for e in moe_layer.experts:
#     print(e.ln_1._am_expert)

# if isinstance(moe_layer, nn.Linear):
#     std = 0.02  # default GPT init
#     # if hasattr(moe_layer, "_is_expert"):
#     #     print('i am an expert')
#     #     std = 0.01  # ✅ smaller init for MoE experts
#     # else:
#     #         print('not working')
#     print('xxxxx')




# %%
# experimenting with how to use msking to select specific rows from a 2d tensor
# 1. Define the 2D tensor
data = torch.tensor([[1, 2, 3],
                     [4, 5, 6],
                     [7, 1, 9],
                     [10, 1, 12]])

print(f"Original data tensor:\n{data}")

# 2. Create the boolean mask that returns true if any element in a row of data is ==1
mask = (data==1).any(dim=-1)
print(f"\nBoolean mask:\n{mask}")

# 3. Apply the mask to all rows and select all columns
selected_rows = data[mask,:]
print(f"\nSelected rows:\n{selected_rows}")

# Apply the mask to all rows and select column 0
selected_rows_col = data[mask,0]
print(f"\nSelected rows col 0:\n{selected_rows_col}")

# select the second and third column of the masked data. 
selected_rows_col = data[mask,1:3]
print(f"\nSelected rows col 1-3:\n{selected_rows_col}")

# create a column wise mask
mask = (data==1).any(dim=0)
print(f"\ncolumn wise Boolean mask:\n{mask}")

# 3. Apply the mask to select cols
selected_cols = data[:,mask]
print(f"\nSelected cols:\n{selected_cols}")


# %%
D ={
    '1' : "exper dummy",
    '3' : "exper dummy",
    '5' : "exper dummy",
    '7' : "exper dummy",
}
B=3
T=5
expert_id_global = torch.tensor([1, 3, 5, 7])
top_k_lcl_indices = torch.tensor([
                                [[2,3],[1,0], [3,1], [2,1], [3,0]],
                                 [[1,2],[1,3], [0,1], [2,0,], [3,2]],
                                 [[1,3],[2,0], [3,3], [1,0], [0,2]]
                                 ])

top_k_global_indices = expert_id_global[top_k_lcl_indices]

print(f'top_k_global_indices\n{top_k_global_indices}\n')
# print(f'{top_k_global_indices.view((B*T), -1)}\n')
print(f'{top_k_global_indices.view(-1)}\n')

for id_str in D.keys():
    id = int(id_str)
    # print(id.shape)
    print(top_k_global_indices.shape)
    print(id)
    expert_mask = (top_k_global_indices == id).any(dim=-1)
    print(expert_mask)
    flat_mask = expert_mask.view(-1)
    print(flat_mask)
    break



# %%
D ={
    '1' : "exper dummy",
    '3' : "exper dummy",
    '5' : "exper dummy",
    '7' : "exper dummy",
}

for i , key in enumerate(D):
    print(i, key)
# %%
print(torch.zeros(5, dtype=torch.int16))

# %%
accum = torch.zeros(13, dtype=torch.int16)
print(f'\naccum:\n{accum}\n')
f = torch.tensor([[1, 2, 3],
                     [4, 5, 6],
                     [5, 1, 5],
                     [10, 1, 12]])

f = torch.flatten(f)
print(f'\ndata_flat:\n{f}\n')
bin_count = torch.bincount(f, minlength=13)
print(f'\nbin_count:\n{bin_count}\n')
accum += bin_count

print(f'\naccum updated:\n{accum}\n')



# %%
num_experts = 32
world_size = 8
rank = 0
experts_per_gpu = num_experts // world_size

for rank in range(8):
    start = (rank * experts_per_gpu)
    end = start + experts_per_gpu -1
    print(f'rank: {rank}  ids {start} - {end}')

for rank in range(8):
    start = (rank * experts_per_gpu)
    end = (rank+1) * experts_per_gpu
    print(f'rank: {rank}  ids {start} - {end}')
# %%
import torch
import torch.distributed as dist
import os

dist.is_initialized()
int(os.environ.get('RANK', -1)) != -1
d =(os.environ.get('RANK'))
print('in distributed') if os.environ.get('RANK') is not None  else print('not')

# %%

import torch
import torch.nn.functional as F

# experiment with how to implement all-to-all communication in PyTorch for MoE layers

num_experts = 16
B=3
T=5
k=2

experts_on_this_gpu_start = 4
experts_on_this_gpu_end = 8
torch.manual_seed(42)

k2 = torch.randint(0,16,size=(B,T,2))
print(print(f'topk indices shape: {k2.shape}\n{k2}\n'))
k2_flat = k2.view((B*T),-1)
print(f'topk indices flat1 shape: {k2_flat.shape}\n{k2_flat}\n')

expert_mask1 = ((k2_flat>=experts_on_this_gpu_start) &(k2_flat<experts_on_this_gpu_end)).any(dim=-1)
print(f'expert_mask1:\n{expert_mask1}')
flat_positions1 = torch.where(expert_mask1)[0]
print(f'\nflat_positions 1: {flat_positions1}\n')


expert_assignments_for_tokens = []
for token_idx in flat_positions1:
    token_experts = k2_flat[token_idx]
    print(f'token_experts: {token_experts}')
    # Filter to only experts on this GPU and convert to local indices
    gpu_expert_mask = ((token_experts >= experts_on_this_gpu_start) & 
                        (token_experts < experts_on_this_gpu_end))
    print(f'gpu_expert_mask: {gpu_expert_mask}')
    token_local_experts = token_experts[gpu_expert_mask] - experts_on_this_gpu_start
    print(f'token_local_experts: {token_local_experts}\n')

    expert_assignments_for_tokens.append(token_local_experts)

print(f'expert_assignments_for_tokens:\n{expert_assignments_for_tokens}\n')
print(f'flat_positions1:\n{flat_positions1}\n')
print('\n')


# code from claude for padding and masking expert token assignments
padded_expert_ids = []
expert_masks = []
for token_expert_ids in expert_assignments_for_tokens:
            current_k = len(token_expert_ids)
            
            if current_k < k:
                # Pad with -1 (invalid expert ID)
                padding = torch.full((k - current_k,), -1, 
                                   device=token_expert_ids.device, dtype=token_expert_ids.dtype)
                padded_ids = torch.cat([token_expert_ids, padding])
                
                # Create mask: True for valid experts, False for padding
                mask = torch.cat([
                    torch.ones(current_k, device=token_expert_ids.device, dtype=torch.bool),
                    torch.zeros(k - current_k, device=token_expert_ids.device, dtype=torch.bool)
                ])
            else:
                padded_ids = token_expert_ids[:k]  # Truncate if somehow > k
                mask = torch.ones(k, device=token_expert_ids.device, dtype=torch.bool)
            
            padded_expert_ids.append(padded_ids)
            expert_masks.append(mask)

print(f'padded_expert_ids:\n{padded_expert_ids}\n')
stacked_expert_ids = torch.stack(padded_expert_ids)
print(f'stacked_expert_ids:\n{stacked_expert_ids}\n')
print(f'expert_masks:\n{expert_masks}\n')



# my code for padding and masking
padded_expert_ids3 = []
for a2 in expert_assignments_for_tokens:
    print(a2)
    if len(a2) < k:
        p3 = F.pad(a2, (0, k-len(a2)), mode='constant', value=-1)
        padded_expert_ids3.append(p3)
    else:
        padded_expert_ids3.append(a2)
padded_expert_ids3_stacked = torch.stack(padded_expert_ids3)
print(f'\npadded_expert_ids3:\n{padded_expert_ids3_stacked}\n')

expert_masks3 = (padded_expert_ids3_stacked >=0)
print(f'\nexpert_masks3:\n{expert_masks3}\n')



# # example input tensor, flatten and mask uisng flat_postions1 as mask
# x_test =torch.rand(3, 8, 7)
# x_test_flat = x_test.view((3*8), -1)
# print(x_test_flat)
# print('\n')
# print(x_test_flat[flat_positions1])


# %%
torch.manual_seed(42)

# experiment with torch scatter and top-k gating. obtaining the topk experts and weights for each token.


logits_noisy = torch.rand(9,8) # #(B*T, num_experts) emulate the output of the linear layer of topk gating with noise added
print(f'logits_noisy:\n{logits_noisy}\n')

top_k_logits, top_k_ids = logits_noisy.topk(2, dim=-1)

print(f'top_k_logits:\n{top_k_logits}\n')

print(f'top_k_ids:\n{top_k_ids}\n')

zeros = torch.full_like(logits_noisy, float('-inf')) 
# print(f'zeros:\n{zeros}\n')

sparse_logits = zeros.scatter(-1, top_k_ids, top_k_logits)

print(f'sparse_logits:\n{sparse_logits}\n')

weights = F.softmax(sparse_logits, dim=-1)
print(f'weights:\n{weights}\n')


token_idx = 2 # assume we are looping over all tokens with one or both of its experts on a given gpu and we are at token_idx

token_experts = top_k_ids[token_idx] # get the topk experts for this token
print(f'\ntoken_experts for token {token_idx}: {token_experts}\n')
# Filter for experts on this GPU

gpu_token_expert_mask = ((token_experts >= 0) & 
                    (token_experts < 4))

print(f'gpu_token_expert_mask: {gpu_token_expert_mask}\n')

token_experts_on_this_gpu = token_experts[gpu_token_expert_mask]
print(f'\ntoken_experts_on_this_gpu: {token_experts_on_this_gpu}\n')


token_weights = weights[token_idx][token_experts]
print(f'token_weights: {token_weights}\n')

token_weights_on_this_gpu = token_weights[gpu_token_expert_mask]
print(f'token_weights_on_this_gpu: {token_weights_on_this_gpu}\n')


# experiment with 


# %%
# creating the tensors for RoPe 
seq_len =12
num_heads = 4
n_embed = 32
B = 2

r = torch.arange(768).reshape(B,seq_len, num_heads, n_embed//num_heads)
print(r[:,:,0,:])
x1 = r[:, :, :, 0: :2]
x2 = r[:, :, :, 1: :2]

print(x1[0,:,0,:])
print(x2[0,:,0,:])


# %%
print(torch.empty(0,2))
print(torch.empty(0,2).numel())
# %%
zer = torch.zeros(4)
print(zer)
zer[1] += 2
print(zer)
# %%
dic = {0: [2,3,5], 1: [9,3,6,7], 4: [5,8]}

for i in range(10):
    
    if i in dic:
        print(f'gpu {i} has positions:{dic[i]}\n')
# %%
# experiment with placing processed tokens back in their original positions
import torch
torch.manual_seed(42)
x_flat = torch.arange(70).view(10,7).float()
print(f'x_flat:\n{x_flat}\n')

output = torch.zeros_like(x_flat)
print(f'output:\n{output}\n')
positions = torch.tensor([1,3,7,9])

processed_tokens = torch.randn(4,7)
print(f'processed tokes:\n{processed_tokens}\n')

output[positions] = processed_tokens
print(f'output after placing processed tokens:\n{output}\n')
# %%
li = []
f = 6
li.append(1)
li
# %%
send_counts = [16, 64]

input_split_sizes_tensor = torch.tensor(send_counts)
output_split_sizes_tensor = torch.empty_like(input_split_sizes_tensor)
input_split_sizes_tensor
output_split_sizes_tensor

# %%

torch.manual_seed = 42
t1 = torch.rand(5,7)
t2 = torch.rand(8,7)
l1 = [t1,t2]

t3 = torch.cat(l1, dim=0)
t3
# %%
t5 = torch.empty(0, 7, device='cpu')
t5
t5.device
# %%
# the softmax of an already softmaxed tensor produces a 
t6 = torch.tensor([4.0,3.0,2.0,2.0])
sf = F.softmax(t6, dim=0)
print(sf)
sf2 = F.softmax(sf)
print(sf2)

# %%
10/2

# %%
