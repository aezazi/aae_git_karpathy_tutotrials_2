#%%
"""
This code takes the bigram model we developed and modifies and echances it to incorporate
attention. I will try and note where there are major changes from the bigram model

Note that in the tutorial, Karpathy is a bit inconsistent and confusing with his
variable naming. In some places he uses (B, T, C) as tensor shape where B is number 
of batches, T ( which he refers to as time) is the number of tokens, and C (which
he refers to as channels) is the dimensions of the embedding for the token.

In other places, as in defining the hyperparameters, he uses n_embd for dimensions
"""
#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

#%%
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# %%
# define hyperparameters
batch_size = 16
block_size = 1024 # sequence length or number of tokens
max_iters = 5000

eval_interval = 100 #number of iterations during training at which we evaluate how model is doing
learning_rate = 1e-3
eval_iters = 200
n_embd = 128 # dimension of the input embedding. Karpathy uses C for channels elsewhere in the code
n_head = 8
n_layer = 6
dropout = 0.12

torch.manual_seed(1337)

# %%
# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# read training text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# print(f'length of input text: {len(text)}')

# %%
# the first 1000 characters
# print(type(text))
# print(text[:1000])

# %%
# get the unique characters by using the set operator
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'number of unique chars: {vocab_size} \n{chars}')

# %%
"""
create a  mapping from characters to integers  and from chars back to integers as a
form of simple tokenaztion
"""
# 
stoi = {ch:i for i, ch in enumerate(chars)} 
# print(stoi)
itos ={i:ch for i, ch in enumerate(chars)}
# print(itos)

# %%
# now create an encoder and decoder using the above mappings
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#%%
t = 'i want to eff leila'
print(encode(t))
print(decode(encode(t)))

# %%
# encode the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(f'data shape: {data.shape}  data type: {type(data)}')
print(data[:100])

# %%
# split the data into training and validation set
n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]
# %%
# create training batches
def get_batch(split):
    data = train_data if split == 'train' else val_data

    # get random starting index for the start of a sequence to pull from the data
    starting_indexes = torch.randint(len(data)-block_size, (batch_size,))
    
    x = torch.stack([data[i : i+block_size] for i in starting_indexes]) # context batch
    y = torch.stack([data[i+1 : i+block_size+1] for i in starting_indexes]) # target batch
    # print(x[0])
    # print(y[0])
    x, y = x.to(device), y.to(device)

    return x, y


# %%
"""
this is code by karpathy that tracks the loss as training proceeds. Don't fully understand how it works, but what he is doing is that every so often he computes the training loss AND THE VALIDATION LOSS. I hadn't seen this before. Usually, I've seen reporting just the training loss. 
"""

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# # %%
# create a single self attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #register_buffer is used for defining parameters that the optimizer does not need to track
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #shape (B, T, C)
        q = self.query(x) #shape (B, T, C)

        wei = q @ k.transpose(2,1) * C**-0.5 # compute scaled score (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #apply mask (B, T, T)
        
        #note that the application of the softmax here IS NOT FOR INTERPRETING wei AS PROBABLITIES. IT IS A FOR NORMALIZING.
        wei = F.softmax(wei, dim=-1) # apply softmax (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out

#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        #Just instantiate desired number of Heads (implemented in the Head class above) into a list
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        #this is for "projection back into the residual pathway". I don't understand this
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        MultiHeadAttention is simply running input x through the desired number of heads and concatenating the output. concatenation is across C (the dimensions of the embedding). After concatention, the sum along the embedding dimension must equal n_embd for further processing. So the head_size * num_heads must equal n_embd.
        """
        out= torch.cat([h(x) for h in self.heads], dim = -1) 
        out = self.proj(out) # projection back into residual pathway
        out = self.dropout(out)
        return out
#%%
"""
This are simple feed forward linear and activations layers. They are applied after MultiHeadAttention As karpathy describes it, after multiattention, the tokens have gathered information from other tokens, but they "haven't thought about what they found from the other tokens"
"""
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # multiplay by 4 for additional dimensions and computational power
            nn.ReLU(),
            # this last linear is for projection back into residual pathway
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout),
        ) 
        
    def forward(self, x):
        return self.net(x)

#%%
#create blocks
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head #Seems to me there should be error trapping here
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        the x added to both self attention and feedforward are residual connections.
        Refer to Karapthy video around 1:27:00 for a great explanation
        """
        x = x + self.sa(self.ln1(x))
        # print(f'shapex: {x.shape}')
        # x = x + self.sa(x)
        
        x = x + self.ffwd(self.ln2(x))
        return x
    

#%%
"""
my implementation of sinusoidal positional encodeing as a class
refer to https://machinelearningmasterycoma-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/ and https://youtu.be/1biZfFLPRSY?si=TayhW-DfrBjUbzUu
for good tutorials
"""
class PositionalEncodingAAE():
    def __init__(self, n_embd =n_embd, n=10000):
        self.n = n
        self.n_embd = n_embd
       
    
    def get_pos_embds(self, seq_length):

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
        denominator = torch.tensor([torch.pow(n, 2*i/self.n_embd)  for i in torch.arange(int(self.n_embd/2))])
        

        """
        create an empty tensor of shape (seq_length x model_dim)
        divide position by the denominator and apply sin to odd number columns and cosine to
        even number columns
        """
        self.pos_embds = torch.zeros(seq_length, self.n_embd)
        self.pos_embds[:,0::2] = torch.sin(k/denominator)
        self.pos_embds[:,1::2] = torch.cos(k/denominator)
        return self.pos_embds.to(device)


# %%
class BigramLanguageModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (vocab_size x embedg dim)
        
        #In the tutorial, he doesn't expalin at all how this position encoding works
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)


        self.position_embedding_table = PositionalEncodingAAE()

        """
        self attention head using Head class above for just one head. This the first step in the
        tutorial showing how single attention head is implemented. It is superceded below by 
        multi-attention and then finally by an attention block
        """
        # self.sa_head = Head(n_embd) 
        
        """
        MultiHeadAtttention with head_size = n_embd // n_head.
        When concatentated, the final output will be 32 dimensional which matches n_embd
        Both sa_heads and ffws will later be incorprated by a block structure. 
        """
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) 
        # self.ffwd = FeedForward(n_embd) #feedforward added later in the tutorial

        """
        So here is the block
        """
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx is the encoded inputs. B is batch size. What he calls T for time, 
        he also refers to as block_size or sequence length; these all mean the number of tokens.
        Both inputs and targets have dimensions (B x T)
        The embedding table orgainzes the idx input of shape B x T into B X T X C where
        C is the embedding dimension.
        """
        B, T = idx.shape
        # print(f'T is: {T}')

        # embed tokens
        tok_emb = self.token_embedding_table(idx) #embed tokens ---> shape (B, T, C)

        """
        The position embedding tensor has shape (T x C) or (sequence lgth x embedding dimension)
        The token and and position embeddings are added to produce the final representation
        that captures both word sense and position. This is consistent with the literature.
        The shape of the position embedding is (T X C) which is elementwise added to the
        token embedding with shape (B X T X C) by broadcasting (T x C) to all B 
        """
        sin_pos_emb =  self.position_embedding_table.get_pos_embds(T)

        # print(f'tok_emb.shape: {tok_emb.shape}')
        # print(f'sin_pos_emb.shape: {sin_pos_emb.shape}')

        x = tok_emb + sin_pos_emb #(B, T, C)

        """
        code below is for applying a single attention head. This will be changed later in
        the tutorial to a block of attention.
        """
        # x = self.sa_head(x) #apply one head of self attention. (B, T, C) superceded by self.sa_heads for multiatttention
        # x = self.sa_heads(x) #apply multiHeadAttention. (B, T, C) later incorporated into self.blocks()
        # x = self.ffwd(x) # apply feed forward. (B, T, C) later incorporated into self.blocks()
        
        x = self.blocks(x) # process the input through the blocks #(B, T, C)
        x = self.ln_f(x) # apply final layer norm
        
        logits = self.lm_head(x) #then apply a linear layer for final output(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            """
            pytorch cross_entropy loss expects (inputs x classes) and (target x classes).
            so the input B x T tensors have to be reshaped. We are reshaping these by 
            stacking the sequence from the 4 batches into one dimension
            """
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            # print('i am here')

        return logits, loss

    def generate(self, idx, max_new_tokens):
        print(self)
        """
        This method is used only in generation not training.
        idx is the input. the code below gets predictions based on the last token in the sequence
        But now, that prediction is informed by the attention head in the forward method above.
        Note the idx is generated text and it's length grows to max_new_tokens.
        """
        
        for _ in range(max_new_tokens):
            """
            as noted above, the length of idx grows to up to max_new_tokens. So the length will
            exceed the length we used for creating the positional encodings which was block_size. 
            So we need to crop the generated output to be equal to block size. We crop from the
            end of the generated idx.
            """
            idx_crop = idx[:,-block_size:]
            # print(f'idx shape: {idx.shape}')
            # print(f'idx_crop shape: {idx_crop.shape}')

            logits, loss = self(idx_crop)
            # print(loss)

            """
            code below focuses only on the last token by slicing the last token 
            from the sequence for each batch. the logits for each last token
            have dimensions 65 which is the vocabulary size. when we put the
            logits through softmax, they represent a conditional probability
            distribution of the next character over the vocab of 65 characters.
            REFER TO 12.3.1 IN DEEP LEARNING BY BISHOP
            """ 
            logits = logits[:,-1,:] #shape (B X C)
            # print(f'logits shape: {logits.shape}')

            """
            code below randomly picks a token from the distribution as the next token. 
            This is a very simplistic form of sampling from the probability distribution.
            REFER TO 12.3.2 IN DEEP LEARNING BY BISHOP re. other techinques like using
            Temprature to modify the logits that are input to softmax.
            """
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # print(f'idx_next shape: {idx_next.shape}')
            # print(f'idx shape: {idx.shape}')
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

model = BigramLanguageModel()
m = model.to(device)

#print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


# %%
# create optimizer and train
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

     # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") 

    context_batch, target_batch = get_batch('train') # get context and target batches

    logits, loss = m(context_batch, target_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# %%
#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


# %%
