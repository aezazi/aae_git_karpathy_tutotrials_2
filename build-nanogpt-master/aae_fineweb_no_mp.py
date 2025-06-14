#%%
"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

from datasets import load_dataset
import tiktoken
import numpy as np
from tqdm import tqdm
import os
import time

#%%
# Constants
tokenizer_name = "gpt2"
shard_size = 100_000_000
shard_dir = "aae_token_shards_no_multiprocess"
os.makedirs(shard_dir, exist_ok=True)

#%%

encoder = tiktoken.get_encoding(tokenizer_name)
eot_token_id = encoder.eot_token # get gpt2 tokenizer eot token id
print(eot_token_id)
#%%
# Load the dataset with streaming to avoid memory overflow
dataset_iterator = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)


#%%
dataset_iterator_test = dataset_iterator.select(range(3))

#%%
# create tokenization function
# NOTE: The use of encoder.encode_ordinary(). This is beacuse When preparing data for training: You usually want full control over where and when special tokens (like <|endoftext|>) are inserted. encode_ordinary gives you just the raw tokenization, so you can insert special tokens like eot_token_id manually, e.g., at shard boundaries or document ends.
def tokenize(example: str, eot=eot_token_id):
    text = example['text']
    tokens = encoder.encode_ordinary(text)
    tokens.append(eot)
    tokens = np.array(tokens, dtype=np.uint16)
    # print('tokeninzing')
    return tokens


#%%
test = dataset_iterator[0]['text']
print(test)
test_tokenize = tokenize(dataset_iterator[0])
print(test_tokenize)
print(len(test_tokenize))
print(len(tokenize(dataset_iterator[1])))
print(len(tokenize(dataset_iterator[2])))

# # test tokenize function
# for example in dataset_iterator:
#     example_text = example['text']
#     print(example_text[0:100])
#     tokens = tokenize(example)
#     print(tokens[:20])
#     print(tokens[-1])
#     print(type(tokens))
#     print(tokens.size)
#     break

# %%

def create_shards(dataset_iterator, shard_dir=shard_dir):
    shard_idx = 0
    # initialize shard tokens with eot token at the begining to signal that this document is a continuation
    shard_tokens_buffer = np.empty(shard_size, dtype=np.uint16)
    shard_tokens_buffer[0] = eot_token_id
    print(f'shard buffer len: {len(shard_tokens_buffer)}')
    shard_token_count = 1

    d_start = time.time()
    for i, example in enumerate(dataset_iterator):
        
        tokens = tokenize(example)
        # print(f'shard idx = {shard_idx} | example: {i} | tokens lenght; {len(tokens)}')

        # my approach differs from Karpathy in that I do not split documents between shards. If current example does not fit in the current shard, save the shard and start a new one
        if shard_token_count + len(tokens) > shard_size:
            shard_save_path = os.path.join(shard_dir, f'shard_{shard_idx:04d}')
            # shard_tokens = np.array(shard_tokens, dtype=np.uint16)
            shard_tokens_final = shard_tokens_buffer[:shard_token_count].astype(np.uint16)
            np.save(shard_save_path, shard_tokens_final)
            
            # measure time to create this shard
            d_end = time.time()
            dt = d_end - d_start
            d_start = d_end

            print(f'saved to {shard_save_path} with {len(shard_tokens_final):,} tokens | time to create shard: {dt:.3f}')

            # start new shard
            shard_idx += 1
            shard_tokens_buffer = np.empty(shard_size)
            shard_tokens_buffer[0] = eot_token_id
            shard_token_count = 1

        shard_tokens_buffer[shard_token_count : shard_token_count+len(tokens)] = tokens
        shard_token_count += len(tokens)
        # print(f'concacting example: {i} to shard: {shard_idx}')
       
        
    if shard_token_count > 0:
            shard_save_path = os.path.join(shard_dir, f'shard_{shard_idx:04d}')
            shard_tokens_final = shard_tokens_buffer[:shard_token_count].astype(np.uint16)
            np.save(shard_save_path, shard_tokens_final)
            
            d_end = time.time()
            dt = d_end - d_start
            print(f'saved to {shard_save_path} with {len(shard_tokens_final):,} tokens | time to create shard: {dt:.2f} secs')

# %%
create_shards(dataset_iterator)
# %%
tokens = np.load("aae_token_shards_no_multiprocess/shard_0000.npy")
print(type(tokens))
print(tokens[0])
print(len(tokens))

# %%
