#%%
#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torchtune
#%%
"""
code below shows how to use matrix multiplication to create a mask that
masks out future tokens and averages all previous tokens
"""
torch.manual_seed(42)
a = torch.ones(3,3)
b = torch.randint(0,10,(3,2)).float()

#this is just a straight matrix multiplication
c = a @ b #matrix multiply a and c
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
# %%
"""
This code uses torch.tril to create a lower triangular matrix. This zeros out future 
tokens from being included in the computation
"""
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

# %%
"""
if we want to use matrix multiplication to produce a final matrix where each element is
the average of the elements that came before it in the same dimension, we normalize 
each row of the lower triangular matix
"""

torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))

"""
torch.sum(a,1, keepdim=True) computes how many non-zero elements in each row of the
upper triangular matrix. keep_dim=True returns the result in a column vector (rather than
the default row vector). Then we can just divide the original upper triangular matrix by
this "normalizing" denominator
"""
print(f'normalizing denominator:\n{torch.sum(a,1, keepdim=True)}')
a = a / torch.sum(a,1, keepdim=True)
b = torch.torch.randint(0,10, (3,2)).float()
c = a @ b 
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')

"""
If we examine c, we see that in the first column, second row, 4 is the average of b first
column first row anf first columns second row (2+6)/2 = 8. similarly, c third row first
column is (2+6+6)/3 = 4.667. The same applies to second column (dimension). So the
normalized a matirx, when multiplied by b, computes the cumlative running average of b
aling each dimension of b
"""
print('c=')
print(c)

# %%
"""
another very clever way to do the normaliztion above is to use softmax on a matrix of
zeros for the elements we want to keep and -infinity for the future elements. I implement
this slightly differently than Karpathy. The main this is that zero elememts in the 
upper triangular matrix have to be set to -infinity 
"""
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))

a = torch.where(a==0, float('-inf'), 1.0)
print(a)

a = F.softmax(a, dim=1)
print(a)



# %%
"""
Now apply the concepts from above to create a masked weights matrix that's initialized
with 0s
"""
T = 3 # sequence length
tril = torch.tril(torch.ones(T,T)) # upper triangular matrix of 1s and 0s
wei = torch.zeros(T,T) #weight matrix initializes with 0s


"""
apply the upper triangular mask to wei, replacing the 0s in the mask with -infinity,
so we end up with a lower triangular matrix with -infinity in the upper part and 0s
in the lower part
"""
wei = wei.masked_fill(tril==0, float('-infinity')) 
print(wei)

"""
finally apply softmax to rows to create a matrix that will take the cumlative running average
of any matrix that it is right multiplied with.
"""
wei = F.softmax(wei, dim=1) 
print(wei)

# %%
"""
the matrix we developed above for computing a cumulative running average of a matrix it is
right multiplied with is the implementation concept for transformer attention. You can
do weighted aggregations of past tokens. this is how we develop self attention blocks
"""

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x[0])

tril = torch.tril(torch.ones(T,T)) 
wei = torch.zeros(T,T)
wei = wei.masked_fill(tril==0, float('-infinity')) 
print(wei)
wei = F.softmax(wei, dim=1) 
print(wei)

xbow = wei @ x
print(x[0])
print(xbow[0])

# %%
"""
the code below is an example of a self attention block
"""
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch, time, channels(dimensions)
x = torch.randn(B,T,C)

# a single self-attentiion head
head_size = 16 # the dimesion of the attention head

#this is a linear layer that transforms the input with dimensions C to dimensions head_size
key = nn.Linear(C, head_size, bias=False)  
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # shape now (B, T, head_size)
q = query(x) # shape now (B, T, head_size)


wei = q @ k.transpose(2,1) #(B, T, head_size) @ (B, head_size, T) --> (B, T, T)

wei_scaled = wei/(head_size**0.5) # scale by dividing by square root of head_size


tril = torch.tril(torch.ones(T,T))
wei_m = wei.masked_fill(tril[:T, :T]==0, float('-inf')) #apply mask

"""
wei_s are the attention coefficients. 
Note that we are taking the softmax across the last dimension of wei shape (B, T, T). This
is normalizing each row.
"""
wei_s = torch.softmax(wei_m, dim=-1)

"""
next we scale
"""
print(f'wei_s masked:\n {wei_s[0]}')

"""
we now multiply the attention coeffecient (or score in some literature) by the value
vector to modify the weights of the input vector x based on the attention scores
"""
v = value(x)
out = wei_s @ v

out.shape


# %%
# here I'm trying to understand transposing of 3d tensor for the  tensor multiplication above
torch.manual_seed(43)
t_o= torch.randint(0,10,(2,3,4))
print(t_o)

t_trans1 = t_o.transpose(2, 1)
print(t_trans1.shape)
print(t_trans1)

# this is how Karpathy indexes, its the same result, but I don't like the negative indexing
t_trans2 = t_o.transpose(-2, -1)
print(t_trans2)

t_n = torch.randint(0,10,(2,5,4))

print(t_n @ t_trans1)



# %%
# experimenting with trying to replicate normalizing q@k_T
torch.manual_seed(1337)

hd = 10
a = torch.randn((2, 3, hd))
b = torch.randn((2, 3, hd))

print(a.var())
print(b.var())

tril = torch.tril(torch.ones(3,3))


c = c.masked_fill(tril==0, float('-inf')) #apply mask
c = a @ b.transpose(-2,-1)
print(c.var())

c = c* hd**-0.5
print(c.var())

# print(torch.sum(torch.var(c, dim=1))/6)
# %%
print(torch.arange(7))

# %%
#%%
#aae test code to understand nn.Embedding and positional embedding
torch.manual_seed(55)
sq_len = 9
v_size = 30
embd_dim =5
tok_embed = nn.Embedding(v_size, embd_dim)
pos_embed = nn.Embedding(sq_len, embd_dim)
print(pos_embed)
tokens_list = [[2, 0,5,9, 11, 13, 8, 2, 7], [4, 0,15,4, 18, 1, 18, 12, 17]]
print(f'length of tokens list: {len(tokens_list)}')
tokens = torch.tensor(tokens_list)
print(f'tokens shape: {tokens.shape}')

te = tok_embed(tokens) #embed tokens
pe = pos_embed(torch.arange(sq_len)) #embed positions
print(f'te shape: {te.shape}')
print(te)
print(f'pe shape: {pe.shape}')
print(pe)

#add token embedding with position embedding
x = te + pe
print(f'x (te + pe) shape: {x.shape}')
print(x)



# print(logits[1])
# s = nn.functional.softmax(logits[1])
# print(s)
# torch.argmax(s)
# %%
# experimenting with negative indexing
b=5
u = torch.tensor(([0,1,2,3,4,5,6,7,8,9,10,11,12,13], [0,1,2,3,4,5,6,7,8,9,10,11,12,13]))
print(u[:, -b:])

# %%
#experimenting with nn.Linear
m = nn.Linear(20, 30)
input = torch.randn(4, 128, 20)
output = m(input)
print(output.size())

# %%
# experimenting with slicing 3d tensor from 4d tensor along diffetent dimensions
batch_size = 2
sequence_length = 12
num_heads = 4
head_dim = 16

tensor_4d = torch.randn(batch_size, sequence_length, num_heads, head_dim)

# Slice a single 3D tensor from the batch (e.g., the first image)
tensor_3d = tensor_4d[:, :, 0, :]
print(tensor_3d.shape)

# %%
