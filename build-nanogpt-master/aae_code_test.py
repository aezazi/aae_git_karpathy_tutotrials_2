#%%
import os
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
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
