#%%
import os
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import aae_utils
# %%
"""
understand decorators. Decorators take a function as the argument, add some additional functionality to that function, and return the modified function with same name as the function.. In that sense, they are like a closure.

Python decorators are a specific application of closures. A closure is a function that retains access to variables from its enclosing scope even after that scope has finished executing. Decorators, in Python, are implemented using closures to wrap and modify the behavior of other functions or methods.
A decorator function typically defines an inner function (the closure) that wraps the original function. This inner function can then access and modify the arguments and behavior of the original function before or after calling it. The decorator function then returns the inner function, effectively replacing the original function with the modified version.
"""
def aae_decorator(func):
    print(f'now in decorator')
    def decorate_func(*args, **kwargs):
        print(f'now inside the wrapprer')
        result = func(*args, **kwargs)
        print('executed the function')
        return ( f'the result of adding {args} is {result}')
    return decorate_func

# %%
# apply the decorator to sum_func function
@aae_decorator
def sum_func(*args):
    return sum(args)

#%%
sum_func(1,2,3)

# %%
# In this section, I'm trying to fully understand nn.embedding
torch.manual_seed(4)

"""
Define an embedding layer for a num_embeddings (vocabulary size) of 10, with embedding_dims (dimenions or features) of 6. note that the 10 defines the integer range 0-9 in the input tensor, not the input tensor size. Each integer 0-9 is the token_id of an input
"""

embedding_layer = nn.Embedding(10, 6)

# this prints out the full table created by the above
input_indices = torch.tensor([0,1,2,3,4,5,6,7,8,9]) 
print(f'full embedding table:\n{embedding_layer(input_indices)}')
print('-'*50)

# rows of the table can be accessed via the row index. Here we access rows 1, 5, 8
input_indices = torch.tensor([1, 5, 8, 5])  # Example token_id indices
print(f'embeddings at index  1, 5 , 8 , 5:\n{embedding_layer(input_indices)}')
print('-'*50)


# this throws an error because index 10 is out of range. It exceeds our vocab size.
input_indices = torch.tensor([1, 2, 10, 9, 5])  
print('accessing 1, 2, 10, 9, 5 throws an error because 10 is out of range and exceeds the table size')
try:
    print(embedding_layer(input_indices))
except IndexError:
    print('\nIndex out of range')


# %%
"""
Now lets the take the embedding table we created above and use it to encode batches of tokens.
First, another print of the table
"""
torch.manual_seed(4)

input_indices = torch.tensor([0,1,2,3,4,5,6,7,8,9]) 
print(f'full embedding table:\n{embedding_layer(input_indices)}')
print('-'*50)

# a batch of 3 sequences, each sequence is of length 4. Each element of each sequence is a token_id with ids in the range 0-9, our vocab size.
ids = torch.randint(10,(3,4))
print(f'the token_ids are ({ids.shape}):\n{ids}')
print('-'*50)


"""
note that ids tensor has shape (3, 4). Pytorch treats the first dimension as the batches and the second as sequence length (so a sequence of 4 token ids in this case). The dimensionality of each token_id (6 in this case) was defined when we created the embedding_layer object above. So the output of the embedding_layer() when applied to ids, has shape (batch_size x sequence length (number_of_token_ids) x dimensions of token) or (3 x 4 x 6 ). So three batches of 4 token_ids (rows) of dimensionality 6 (columns)

we can make the sequence and batch length whatever we want, but the ids have to be 0-9 since that's the vocabulary size we defined when we created the Embedding object with embedding_layer = nn.Embedding(10, 6) 
"""
print(f'the embeddings of the  token_ids are {embedding_layer(ids).shape}:\n{embedding_layer(ids)}')


#this code outputs 100 batches of 768 token_ids with each id having dimensionality  (100 X 768 x 6)
ids = torch.randint(10,(100,768))
print('-'*50)
print(f'the embeddings of the  token_ids are {embedding_layer(ids).shape}:\n{embedding_layer(ids)}')


# %%
# understant pytorch tensor dimensions
torch.manual_seed(4)

print('creating a random tensor of shape (2, 4, 5)')
t = torch.rand(2,4,5) # pytorch treats the first dimension as the batch dimension
print(t)
print('-'*60)

print('transposing the second and third dimensions of the tensor')
t_t = t.transpose(2,1) #transpose the second and third dimensions
print(t_t)
print('-'*60)

# reshaping the tensor to shape (2, 5, 4). This is as if the tensor was flattened  to a (1, 20) tensor and then reshaped to (5, 4) for each batch
print('reshaping the tensor to shape (2, 5, 4)')
t_r = t.view(-1,5,4)
print(t_r)

#%%
"""
test behavior of reshape vs view. Karpathy prefers view

the major difference is that view requires that the input tensor be contiguous and that the new shape have the exact same number of elements as the input. reshape is more flexible...see docs
"""
# 

# a one dimensional tensor an example of a long tensor of token ids
t1 = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(t1)

# we want to break the above tensor into 3 batches
t1_view = t1.view(3,5)
print(f'\nusing view to create (3x5)\n{t1_view}')

t1_reshape = t1.reshape(3,5)
print(f'\nusing reshape to create (3x5)\n{t1_reshape}')
# %%
# experiment with object references to same memory location
a = [1,2,3]
b = a
c = b
print(id(a), id(b), id(c))

# since c points to the same memory location as a, changing c changes a
c.append(4)
print(a)

# %%
# experiment with 4 dimensional tensors. In the context of transformers, the 1st dimension is the batch size, the second is the number of heads, the third is the sequence length, and the fourth is the head size. The printout below is a 4D tensor with shape (2,3,4,5) where 2 is the batch size, 3 is the number of heads, 4 is the sequence length, and 5 is the head size (dimensions of each head)

t2 = torch.rand(2,3,4,5) 
print(t2)


# %%
# experiment with Pytorch's scaled_dot_product_attention(query, key, value). The docs for the function are here: 
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
# When causal is set to True, an attention mask is applied for us.
# the function also scales the dot product of q@k by 1/sqrt(d_k) where d_k is the dimensionality of the key vectors, then applies softmax to the attention weights, and then dropout. This output is then multiplied by the value tensor to get the final attention output. So in essence,the function does all q@kv calculations and returns the attention output.  We just need to provide the query, key, and value tensors.

# Define the query, key, and value tensors. Not
n_embd = 192 # the dimensionality of the input embeddings
batch_size = 2
n_heads = 3
seq_length = 6
head_size = 64 # the dimensionality of each head
query = torch.randn(batch_size, n_heads, seq_length, head_size)
key = torch.randn(batch_size, n_heads, seq_length, head_size)
value = torch.randn(batch_size, n_heads, seq_length, head_size)

# Calculate the scaled dot product attention
attention_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

# Print the output shape
print(f'shape of output from F.scaled_dot_product_attention: { attention_output.shape}') # Expected output: torch.Size([2, 3, 6, 64])

print(f'shape after attention_output.transpose(1,2)    {attention_output.transpose(1,2).shape}')  # Expected output: torch.Size([2, 6, 3, 64])

attention_output_reshaped = attention_output.transpose(1,2).reshape(batch_size, seq_length, n_heads * head_size)
print(f'shape of output after transposing and reshaping: { attention_output_reshaped.shape}')  # Expected output: torch.Size([2, 6, 192])

# using karpathy's method
attention_output_reshaped_karpathy = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, head_size * n_heads)
print(f'shape of output after transposing and using karpathy method with view: { attention_output_reshaped_karpathy.shape}')  # Expected output: torch.Size([2, 6, 192])

# check if the two reshaped outputs are equal
print(f'are the two reshaped outputs equal? {torch.equal(attention_output_reshaped, attention_output_reshaped_karpathy)}')

attention_output_reshaped == attention_output_reshaped_karpathy

# %%
# trying to understand the self.apply() method in Pytorch. It applies a function to all submodules of a module. In the example below, we use it to initialize the weights of a linear layer using Xavier uniform initialization and set the bias to zero.
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = MyModel()

# %%
# understand reshaping of tensor shape (B, T, C), the final output of the GPT model to (B*T, C) for comparison of logits and labels. 
logits = torch.randn(2, 3, 4)
print(logits)

print(logits.view(-1, logits.size(2)).shape)
print(logits.view(-1, logits.size(-1)))
# %%
logits.view(-1).shape

# %%
# understand how to use torch.gather
t = torch.tensor([[1, 2], [3, 4]])
print(t)
print('-'*60)

 # these are the indices we want to select from t. Note that the first index selects the row and the second index selects the column. [0, 0] selects the 0th index of the first row from t twice. [1, 0] selects the 0th and 1st index from the second row of t. Dimension 0 is the row dimension and dimension 1 is the column dimension. In this case we are selecting along the column dimension (1) from each row of t.
index = torch.tensor([[0, 0], [1, 0]]) 
result = torch.gather(t, 1, index)
print(result)
print('-'*60)


# Note that the result does not have to be the same shape as t. In the case below, we are selecting three elements from each row of t.
index = torch.tensor([[0, 0, 1], [1, 0, 1]]) 
result = torch.gather(t, 1, index)
print(result)
print('-'*60)

# here we are selecting along the row dimension (0) from each column of t. In the case below, the first column of the result is the 0th index of the first column of t and the 0th index of the second column of t. The second column of the result is the 1st index of the first column of t and the 0th index of the second column of t.
index = torch.tensor([[0, 0], [1, 0]]) 
result = torch.gather(t, 0, index)
print(result)
print('-'*60)



# %%
"""
# experimenting with pytorch's AdamW optimizer and learning rate scheduler
# AdamW is a variant of the Adam optimizer that decouples weight decay from the optimization step. 
# It is particularly useful for training large models with weight decay regularization. The main difference between Adam and AdamW is that in AdamW, the weight decay term is applied directly to the weights before the update step, rather than being included in the gradient calculation.
# This decoupling allows for better control over the weight decay and can lead to improved generalization performance, especially in large-scale training scenarios.
# cosine annealing is a learning rate schedule that gradually decreases the learning rate over time, following a cosine function. It starts with a high learning rate and decreases it to a minimum value, then restarts the cycle. This approach can help the model converge more effectively by allowing it to explore the parameter space initially and then fine-tune the weights as training progresses.
# The CosineAnnealingWarmRestarts scheduler is a variant of cosine annealing that includes warm restarts. It periodically resets the learning rate to a higher value, allowing the model to escape local minima and explore the parameter space more effectively. This can lead to better convergence and improved performance in some cases.
# I have also added a warmup phase to the learning rate schedule. The warmup phase gradually increases the learning rate from a small value to the initial learning rate over a specified number of steps. This can help stabilize training in the early stages and prevent large updates that could lead to divergence.

This is my implenetation of the learning rate scheduler with warmup and cosine annealing with warm restarts and is different than Karpathy's implementation 
"""

#%% creating a class to encapsulate the learning rate scheduler
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
        
        if restart:
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)
        else:
            self.scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, 
                                               eta_min=self.min_lr)

    def set_lr(self, step):
        if step < self.warm_up_steps:
            # Linear warmup: scale up from 0 to max_lr
            warmup_lr = self.max_lr * (step) /self.warm_up_steps
            # print(f'setting warmup lr to {warmup_lr}')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        if step >= self.warm_up_steps:
            # Step the cosine scheduler
            self.scheduler.step(step - self.warm_up_steps)
        
        self.lrs.append(self.optimizer.param_groups[0]['lr']) 

# %%
# test the lr_scheduler class'
total_steps = 50
model= nn.Linear(512, 512)
max_lr = 6e-4
min_lr = max_lr * 0.1
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
sch = CosineLearingRateScheduler(optimizer=optimizer, restart=False, T_max=total_steps, warm_up_steps=10, max_lr=max_lr, min_lr=min_lr, T_0=10)

# %%
for step in range(total_steps):
    sch.set_lr(step)
   
plt.figure(figsize=(10, 5))
plt.plot(range(total_steps), sch.lrs, marker='o')
plt.title('Learning Rate Schedule: Warmup + CosineAnnealingWarmRestarts')
plt.xlabel('step')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()

# %%
print(sch.lrs)

# %%
my_list = [1, 2, 3]
my_iterator = iter(my_list)



# %%
# experimenting with how to construct a numpy array that is exactly the size of the tokens processed. the intent is to create a buffer that can hold the tokens and then slice it to the exact size of the tokens processed. this way documents don't have to be split between seperate shards.
tok_buffer = np.empty(20, dtype=np.uint16)
tok_buffer[0] =256
# print(type(tok_buffer[4]))
tok_count = 1
print(tok_buffer)
print('-'*40,'\n')

a = np.array([1,2,3,4])
tok_buffer[tok_count : tok_count+len(a)] = a
tok_count += len(a)
print(tok_buffer)
print('-'*40,'\n')


b = np.array([5,6,7,8,9,10,11])
tok_buffer[tok_count : tok_count+len(b)] = b
tok_count += len(b)
print(tok_buffer)
print('-'*40,'\n')

tok_final = tok_buffer[:tok_count]
print(tok_final)


# %%
# # testing the DataLoaderShardMultiGPU class from aae_utils.py
# from aae_dataloader_utils import DataLoaderMultiGPU as loader

# l2 = loader(201, 1024, split='train', shard_dir='aae_edu_fineweb10B_shuffle')

# # %%
# import numpy as np
# # l.shards.sort()
# print(l2.shard_files)
# # print(type(l2.shards[0]))
# n = np.load(f'{l2.shard_dir}/{l2.shard_files[0]}')
# print(n.shape, n.dtype)

# # %%
# # this is the shard index for the 9th file in the list of shard files. really clever way to loop back to the first file after reaching the end of the shard files list
# print(len(l2.shard_files))

# 98 % len(l2.shard_files) 
# # %%
# import torch
# def load_tokens(filename):
#     npt = np.load(filename)
#     # npt = npt.astype(np.int32) # added after video
#     ptt = torch.tensor(npt, dtype=torch.int32) # changed to int32 after video
#     return ptt
# ptt = load_tokens(f'{l2.shard_dir}/{l2.shard_files[0]}')
# ptt.dtype, ptt.shape, ptt[:10]
# # %%
# 99 % len(l2.shard_files) # this is the shard index for the 9th file in the list of shard files
# # %%
# os.makedirs('test_dir', exist_ok=True)
# t = np.load(f'{l2.shard_dir}/{l2.shard_files[0]}')
# s_1 = t[:11]
# s_2 = t[10:22]
# s_3 = t[20:32]
# print(s_1)

# %%
# experimenting with methods to save and load data list of lists and arrays to keep each document separate and intact so as to enabale data shuffling when loading the data. 
import tiktoken
import numpy as np
from itertools import chain
shard_list = []
enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
tok1 = enc.encode_ordinary("This is a test of the tiktoken")
tok1 = np.array(tok1, dtype=np.uint16)
tok2 = enc.encode_ordinary("I want to go swimming")
tok2 = np.array(tok2, dtype=np.uint16)
tok3 = enc.encode_ordinary("I am getting old")
tok3 = np.array(tok3, dtype=np.uint16)
# tok1.insert(0, 20566)

print(tok1)
print('-'*40)
print(tok2)
print('-'*40)
shard_list.append(tok1)
shard_list.append(tok2)
shard_list.append(tok3)
print(f'list of arrays of tokens')
print(shard_list)
print(type(shard_list[0]))
print('-'*40)

shard_array = np.array(shard_list, dtype=object)
print(shard_array)
print('-'*40)

np.save("shard_000.npy", shard_array)


loaded = np.load("shard_000.npy", allow_pickle=True)
print(f'loading and printing from saved object')
print(loaded)
print('-'*40)
np.random.shuffle(loaded) 
print(f'loaded 0:\n{loaded[0]}')
print(f'shuffled list of arrays')
print(loaded)
print('-'*40)

loaded[0] = np.concatenate(([eot], loaded[0]))

flat = np.fromiter((x for sublist in loaded for x in sublist), dtype=np.int32)

flat


# %%
t = np.concatenate(([eot], tok1))
t

#%%
from dataclasses import dataclass
import os
import csv
@dataclass
class LogParamConfig:
    # def __init__(self):
    #     pass
    ddp: int 
    ddp_world_size: int =5
    # ddp_local_rank = ddp_local_rank
    ddp_rank: int = 1
    model: object = 'model'
    device: str = 'cuda'
    loss_dir = 'test_dir'
    loss_file = 'test_file.csv'

    def __post_init__(self):
        os.makedirs(self.loss_dir, exist_ok=True)
        print('here')
        print(os.path.join(self.loss_dir, self.loss_file))
        print('here2')
        self.loss_file = os.path.join(self.loss_dir, self.loss_file)
        print('here3')
        print(self.loss_file)

        with open(self.loss_file, "w") as f: # open for writing to clear the file
            csv_out = csv.writer(f)
            csv_out.writerow(['step', 'train_loss']) # write the header row

#%%
t = LogParamConfig(ddp=34)
# t.make_file()
print(t.loss_file)
t.ddp

# %%
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('hella_loss/hellaswag_eval_restart.csv')
df1
plt.figure(figsize=(10, 7))

plt.plot(df1['step'], df1['hellaswag_accuracy'])
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('HellaSwag Accuracy Over Steps')
# plt.xticks(rotation=45) # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

plt.figure(figsize=(10, 8))
df2 = pd.read_csv('loss/train_loss_restart.csv')
plt.plot(df2['step'], df2['train_loss'])
plt.xlabel('Step')
plt.ylabel('Train Loss')
plt.title('Train Loss Over Steps')
# plt.xticks(rotation=45) # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# %%
import numpy as np
#experimenting for most effiecient method to store numpy arrays (tokenized docs) in a shard that allows shuffling of each array wihin the shard
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([10, 20, 30, 40, 50])
array3 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

l = [array1, array2, array3]


shard_array = np.array(l, dtype=object) 
np.save('multiple_arrays', shard_array)

array_of_lists = np.load("multiple_arrays.npy", allow_pickle=True)
print(array_of_lists)
print(type(array_of_lists[0]))
np.random.shuffle(array_of_lists)
print(array_of_lists)
len(array_of_lists)

# %%


# %%
# understanding rotary embeddings
# rotrary position embedding compute theta. exploring two approaches
print(f'formula from paper for computing theta:/nθi = 10000−2(i−1)/d, i ∈ [1, 2, ..., d/2]\n')

i_paper = np.arange(1, 13).reshape(1,-1)
d = i_paper.shape[1]
i_paper_for_theta = i_paper[:(d//2)]
print(f'i_paper using 1 based indexing:\n{i_paper}')

print(f'd dimension size: {d}')
print('-'*50)
theta_1 = 10_000 ** (-2*(i_paper_for_theta-1)/d)
print(f'theta using 1 based indexing and the formula from the paper:\n{theta_1}')

print('-'*50)
i = np.arange(0,12)
print(f' i using  the common pracitce 0 based indexing :\n{i}')

print(f'i_commom_practice_for theta:')
print(np.arange(0, d, 2))
theta_2 = 10000 ** (-np.arange(0, d, 2) / d)
print(f'theta using 0 based indexing and the common practice formula:\n theta = 10000^(-2i/dim)')
print(theta_2)
print(f'\nNote that that two approaches yield exactly the same result')



# %%
# rotary position encoding compute position,theta matrix
seq_length = 5
position = np.arange(seq_length).reshape(1,-1)
print(f'position vector:\n{position}\n')

# the position x theta matrix is the outer product of (position, theta). It multiplies each position index by each theta resultin in a (seq_lenght x dim/2) matrix. dim/2 is the lenght of the theta vector. this matrix is known as the frequency
print('the position x theta matrix is the outer product of (position, theta) which is \nmatmul(position, theta_tranpose)\n')

theta_2 = theta_2.reshape(1,-1)
print(theta_2.shape)

freq = np.outer(position, theta_2) # compute freq matrix using outer product

freq_2 = np.matmul(position.T, theta_2) #alternative method for computing freq

print(np.allclose(freq, freq_2)) # both are equivalent, outer probably more efficient

print(freq_2)



#%%
import torch
# Settings

batch_size = 2
seq_len = 5
num_heads = 3
head_dim = 6  # must be even
assert head_dim % 2 == 0

# Input tensor: [B, S, H, D]
x = torch.arange(180).reshape(batch_size, seq_len, num_heads, head_dim)

# Compute rotary angles
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
pos = torch.arange(seq_len, dtype=torch.float)
angles = torch.outer(pos, inv_freq)  # [seq_len, dim /2]


# Apply sin and cos to angles and use unsqueeze to add dimensions to match number of dimensions of input vector 
sin = angles.sin().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]


x1 = x[:, :, :, : :2]
x2 = x[:, :, :, 1: :2]
# print(x1)

# Apply rotation. Note that the elementwise multiplication broadcasts the sin and cos values into batch and num_heads dimensions
x1_rot = x1 * cos - x2 * sin #[B, S, num_heads,  head_dim/2]
x2_rot = x1 * sin + x2 * cos #[B, S, num_heads,  head_dim/2]
print(f' \nshape x1_rot: {x1_rot.shape}')

# Stack into [B, S, head_num, head_dim/2, 2] the dim=-1 adds a new dimension to the end of [B, S, H, head_dim/2] and stacks each corresponding element from dim=1 of [seq_length, dim/2] from the x_rotated_even and x_rotated_odd vectors into that new third dimension
x_rot = torch.stack([x1_rot, x2_rot], dim=-1) #[B, S, H, head_dim/2, 2]
# print(x_rot)

# flatten last two dims back to [B, S, H, head_dim]
x_rot = x_rot.flatten(start_dim=3, end_dim=-1) 
# print(x_rot)
# print(x_test)
# print(torch.allclose(x_rot, x_test))

print("Original shape:", x)
print("After RoPE:", x_rot)

# %%
# explore multidim slicing
v = torch.arange(180).reshape(batch_size, seq_len, num_heads, head_dim)# [seq_len, dim]
s1 = v[:, :, :, : :2]
s2 = v[:, :, :, 1: :2]
print(s1)
print(s2)


# %%
# Test RotaryPosEmbed class
from aae_utils import RotaryPosEmbed 
x = torch.arange(180).reshape(batch_size, seq_len, num_heads, head_dim)
rotation = RotaryPosEmbed(seq_len=5, head_dim=6)
x_rot2 = rotation.apply_rotation(x)
print(x_rot2)
print(torch.allclose(x_rot, x_rot2))


# %%
# exploring chunking shapes
qkv = torch.randint(10, (3,4, 15))
print(qkv.shape)
q, k, v = qkv.chunk(3, dim=-1)
print(q.shape)

# %%
# exploring pytorch broadcasting
A = torch.tensor([[[[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11,12,13,14,15],
                    [16,17,18,19,20]],

                   [[21,22,23,24,25],
                    [26,27,28,29,30],
                    [31,32,33,34,35],
                    [36,37,38,39,40]],

                   [[41,42,43,44,45],
                    [46,47,48,49,50],
                    [51,52,53,54,55],
                    [56,57,58,59,60]]],

                  [[[61,62,63,64,65],
                    [66,67,68,69,70],
                    [71,72,73,74,75],
                    [76,77,78,79,80]],

                   [[81,82,83,84,85],
                    [86,87,88,89,90],
                    [91,92,93,94,95],
                    [96,97,98,99,100]],

                   [[101,102,103,104,105],
                    [106,107,108,109,110],
                    [111,112,113,114,115],
                    [116,117,118,119,120]]]])

B = torch.tensor([[1, 10, 100, 1000, 10000],
                  [2, 20, 200, 2000, 20000],
                  [3, 30, 300, 3000, 30000]])

B = B.unsqueeze(0).unsqueeze(2)
print(A*B)



# %%
