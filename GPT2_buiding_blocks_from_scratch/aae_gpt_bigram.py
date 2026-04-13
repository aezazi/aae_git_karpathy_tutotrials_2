#%%
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# %%
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# %%
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# %%
# print(text)
print(f'length of training datain chars: {len(text)}')
print(text[:1000])
# %%
# created a sorted list of unique characters in the dataset.
# use the set operator to get just the unique characters and eliminate duplicates
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(len(chars))

# %%
#encoder for tokenizing. Map characters to integers as tokens
stoi = {ch:i for i, ch in enumerate(chars)} 
print(stoi)
# reverse key value mapping as decoder from integer tokens back to characters
itos = {i:ch for ch, i in stoi.items()  } 
# print(chars)
# print(stoi)
# print(itos)
encoder = lambda text: [stoi[c] for c in text]
decoder = lambda tokens: ''.join([itos[t] for t in tokens])

# %%
#example of encoding and decoding a phrase
print(encoder('i want to eff Fariba'))
print(decoder(encoder('i want to eff Fariba')))


# %%
# encode the entire dataset
text_encoded = encoder(text)
# put in a pytorch tensor
data = torch.tensor(text_encoded, dtype=torch.long)
print(f'data shape: {data.shape}  data type: {type(data)}  {data.dtype}')
print(data[:1000])

# %%
# do a train test split
percent_train = round(.9 * data.shape[0])
# print(percent_train)
train_data = data[:percent_train]
val_data = data[percent_train:]

# %%
#block size or context size
block_size = 8
train_data[:block_size+1]


# %%
"""
the target for a sequence is the charcter that comes after that sequence
so for the sequence [18, 47, 56, 57, 58,  1, 15, 47, 58], we initiate with 18. The 
target for the next character is 47, then target for the sequence 18, 47 is 56, 
then the target for the sequence 18, 47, 56 is 57 and so on
"""
x = train_data[:block_size] #training inputs
y = train_data[1: block_size+1] # targets

for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(f'context: {context}  target: {target}')

# %%
# now apply the above principle along with a batching
torch.manual_seed(1337)
batch_size = 4 #number of sequences per batch
block_size = 8
i_random = torch.randint(len(data) - block_size, (batch_size,))
print(i_random)

#%%
"""
the code below is my implementation for generating 4 sequences of length 8
and then stacking into a tensor of four rows to construct a batch
"""
torch.manual_seed(1337)
def get_batch_aae(split):
    data = train_data if split=='train' else val_data
    #the code below randomly picks a starting index for each batch sequence in the batch
    i_random = torch.randint(len(data) - block_size, (batch_size,))
    # print(f'i_random is: {i_random}')
    
    x_batch = []
    y_batch = []

    for i in i_random:
        context_sequence = torch.tensor(data[i:i+block_size])
        # print(f'x is: {x}')
        target_sequence = torch.tensor(data[i+1:i+block_size+1])
        # print(f'y is: {y}')
        x_batch.append(context_sequence)
        y_batch.append(target_sequence)
    return x_batch, y_batch

xx, yy = get_batch_aae('train')
x_batch = torch.stack(xx)
y_batch = torch.stack(yy)
print(x_batch)
print(y_batch)

#%%
"""
the code below is Karpathy's much more elegant implementation
"""
torch.manual_seed(1337)
def get_batch(split):
    data = train_data if split=='train' else val_data
    #the code below randomly picks a starting index for each batch sequence in the batch
    #note that you could also use python random integer generators
    i_start_random = torch.randint(len(data) - block_size, (batch_size,)) 

    # note that you could also use python random integer generators:
    # random.sample(range(0, len(train_data)-block_size), batch_size)
    # [random.randint(0, len(train_data)-block_size) for i in range(batch_size)]


    """
    note the order of execution. the list comprehension gets executed first. Inside
    the list comprehension, data[i:i+block_size] creates a sliced list from data
    that is appended to the enveloping list comprehension. so the list comprehension
    produces a list of lists that torch.stack then stacks into a batch_ size x block_size
    tensor
    """
    x_batch = torch.stack([data[i:i+block_size] for i in i_start_random])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in i_start_random])
    return x_batch, y_batch


context_batch, target_batch = get_batch('train')
# print(f'context:\n {context_batch}')
# print(f'target:\n {target_batch}')

#%%
"""
now take each batch and sequence within each batch to generate auto-regressive training
and target inputs
"""
for batch_idx in range(batch_size):
    for token_idx in range(block_size):
        # print(context_batch[batch_idx])
        context = context_batch[batch_idx,:token_idx+1]
        target = target_batch[batch_idx,token_idx]
        print(f'when context is: {context} target is: {target}')
        

#%%
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        """
        a lookup table of vocab_size x dimensions of the embedding. In this case
        the dimensions are also equal to vocab_size. O don't understand why as yet.
        Perhaps because we will do comparison to onehot target later?
        """
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        idx is the inputs. B is batch size, which is 4 in this case . 
        What he calls T for time, is the block or sequence length, in this case 8
        both inputs and targets have dimensions (B x T)
        the embedding table orgainzes the idx input of shape B x T 
        into B X T X C whre C is  embedding dimenesion, in this case vocab_size
        """
        logits = self.token_embedding_table(idx)

        # if no targets are provided return NOne for the loss
        if targets is None:
            loss= None
        else:
            B, T, C = logits.shape #unpack logits shape into variables
            """
            pytorch cross_entropy loss expects (inputs x classes) and (target x classes).
            so the input B x T tensors have to be reshaped. We are reshaping these by 
            stacking the sequence from the 4 batches into one dimension
            """
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            """ 
            code below gets predictions.
            note that I did not know that you can use a construct like self(idx) to refer
            to the instantiation of the class from within the class
            """
            logits, loss = self(idx)

            """
            code below focus only on the last token by slicing the last token 
            from the sequence for each batch. the logits for each last token
            have dimensions 65 which is the vocabulary size. when put the
            logits through softmax, they represent a conditional probability
            distribution of the next character over the vocab of 65 characters.
            REFER TO 12.3.1 IN DEEP LEARNING BY BISHOP
            """ 
            logits = logits[:,-1,:] #this will have shape (B x C)
            probs = F.softmax(logits, dim=-1) #this will also have shape (B x C)

            """
            code below randomly picks a token from the distribution as the next token. 
            This is a very simplistic form of sampling from the probability distribution
    
            """
            idx_next = torch.multinomial(probs, num_samples=1) 
            # print(idx, idx_next)


            """
            code below concats the next token to the orginal input
            """
            idx = torch.cat((idx, idx_next), dim=1) # this will have shape (B x T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(context_batch, target_batch)
print(logits.shape)
print(loss)

"""
torch.zeros((1, 1) initiates the generation process. this is the new line token which is
a reasonable choice for starting the generating process.
Note that the output is rubbissh because we are not training anything. 
"""
print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#%%
print(m.parameters)
#%%
"""
so now let's create an optimizer and do a proper training run
"""
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(10000):
    context_batch, target_batch = get_batch('train')
    logits, loss = m(context_batch, target_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

# m.generate(idx = torch.zeros((1, 1), dtype=torch.long), 
# max_new_tokens=500)

# the results are returned in a (1 x 501) tensor, the fist row contains all the generate tokens
print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), 
max_new_tokens=500)[0].tolist()))



