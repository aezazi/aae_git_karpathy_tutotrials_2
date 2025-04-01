#%%

"""
various implementions of sinusiodal positional encoding I found.
refer to https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
and https://youtu.be/1biZfFLPRSY?si=TayhW-DfrBjUbzUu
"""
import torch
import torch.nn as nn
import math

#%%
n_embd = 6
seq_length = 4

#%%
#implemenat
def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings

# %%
P1 = sinusoidal_positional_embedding(seq_length, n_embd)
torch.set_printoptions(precision=7)
print(P1)
# %%
s = sinusoidal_positional_embedding
s(4,2)

# %%
# this is an implemnetation of positional encoding from Pytorch docs
# I modified ot to get rid of batch dimension
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model = n_embd, max_len: int = seq_length):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, embedding_dim]``
        """
        return self.pe

#%%
pe = PositionalEncoding()
P2 = pe()
P2
#%%
# code implementing position embedding
 
def getPositionEncoding(seq_len, d, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        print(f'k is: {k}')
        for i in torch.arange(int(d/2)):
            print(f'i is :{i}')
            denominator = torch.pow(n, 2*i/d)
            print(denominator)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
        print('-'*40)
            
    
    return P
 
P3 = getPositionEncoding(seq_len=seq_length, d=n_embd, n=10000)

print(P3.shape)
print(P3)


# %%
"""
My implementation
Position encoding results in a tensor with shape (seq_length x model_dim)
"""
def pos_encoding_aae():
    # seq_length = block_size
    # model_dim = n_embd
    n = 10000

    """
    take the sequence length, generate a sequence of intergers starting with 0, then reshape into
    a column vector where each row number corresponds to the kth position in the sequence.
    """
    k = torch.arange(seq_length).reshape(seq_length,1)

    """
    the denominator in the positional encoding formula: n**(2*i/d)
    creates a (1 x n_emb/2) tensor. the range for i is as per the paper. The same ith index
    denominator is used to compute sin(k/denominator) and then cos(k/denominator) intermittently
    """
    denominator = torch.tensor([torch.pow(n, 2*i/n_embd)  for i in torch.arange(int(n_embd/2))])
    

    """
    create an empty tensor of shape (seq_length x model_dim)
    divide position by the denominator and apply sin to odd number columns and cosine to
    even number columns
    """
    pos_embds = torch.zeros(seq_length, n_embd)
    pos_embds[:,0::2] = torch.sin(k/denominator)
    pos_embds[:,1::2] = torch.cos(k/denominator)

    return pos_embds

P4 = pos_encoding_aae()
    

#%%
"""
my implementation as a class
"""
class PositionalEncodingAAE(nn.Module):
    def __init__(self, seq_length = seq_length, n_embd=n_embd, n=10000):
        super().__init__()
    
        """
        take the sequence length, generate a sequence of intergers starting with 0, then reshape into
        a column vector where each row number corresponds to the kth position in the sequence.
        """
        k = torch.arange(seq_length).reshape(seq_length, 1)

        """
        the denominator in the positional encoding formula: n**(2*i/d)
        creates a (1 x n_emb/2) tensor. the range for i is as per the paper. The same ith index
        denominator is used to compute sin(k/denominator) and then cos(k/denominator) 
        """
        denominator = torch.tensor([torch.pow(n, 2*i/n_embd)  for i in torch.arange(int(n_embd/2))])
        

        """
        create an empty tensor of shape (seq_length x model_dim)
        divide position by the denominator and apply sin to odd number columns and cosine to
        even number columns
        """
        self.pos_embds = torch.zeros(seq_length, n_embd)
        self.pos_embds[:,0::2] = torch.sin(k/denominator)
        self.pos_embds[:,1::2] = torch.cos(k/denominator)

    def forward(self):
        return self.pos_embds

P5= PositionalEncodingAAE().forward()
print(P5)

  


# %%
print(torch.allclose(P1, P4))
print(torch.allclose(P1, P3))
print(torch.allclose(P2, P4))
print(torch.allclose(P2, P5))

# %%
"""
experimenting with manipulating torch tensor to compute every other column
Note that k with shape (4 x 1) is broadcast to match shape of n_embd (4 x 6)
"""
# denominator = torch.tensor([torch.pow(n, 2*i/n_embd)  for i in torch.arange(int(n_embd/2))])
# print(denominator)
te = torch.zeros(seq_length, n_embd)
te[:,0::2] = k/10
te[:,1::2] = torch.cos(k/10)
te


# %%
