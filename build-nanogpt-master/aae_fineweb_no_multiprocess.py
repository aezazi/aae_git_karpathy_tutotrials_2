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
dataset_iterator = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=True)

#%%
# create tokenization function
# NOTE: The use of encoder.encode_ordinary(). This is beacuse When preparing data for training: You usually want full control over where and when special tokens (like <|endoftext|>) are inserted. encode_ordinary gives you just the raw tokenization, so you can insert special tokens like eot_token_id manually, e.g., at shard boundaries or document ends.
def tokenize(example: str, eot=eot_token_id):
    text = example['text']
    tokens = encoder.encode_ordinary(text)
    tokens.append(eot)
    return tokens

# test tokenize function
# for example in dataset_iterator:
#     example_text = example['text']
#     print(example_text[0:100])
#     tokens = tokenize(example)
#     print(tokens[:20])
#     print(tokens[-1])
#     break

# %%

def create_shards(dataset_iterator, shard_dir=shard_dir):
    shard_idx = 0
    # initialize shard tokens with eot token at the begining to signal that this document is a continuation
    shard_tokens = [eot_token_id] 
    shard_token_count = len(shard_tokens)

    d_start = 'd'+f'{shard_idx}' 
    d_start = time.time()
    for example in dataset_iterator:
        tokens = tokenize(example)

        # my approach differs from Karpathy in that I do not split documents between shards. If current example does not fit in the current shard, save the shard and start a new one
        if shard_token_count + len(tokens) > shard_size:
            shard_save_path = os.path.join(shard_dir, f'shard_{shard_idx:06d}')
            np.save(shard_save_path, shard_tokens)
            
            # measure time to create this shard
            d_end = 'd'+f'{shard_idx+1}' 
            d_end = time.time()
            dt = d_end - d_start
            d_start = d_end

            print(f'saved to {shard_save_path} with {len(shard_tokens):,} tokens | time to create shard: {dt:.3f}')

            # start new shard
            shard_idx += 1
            shard_tokens = [eot_token_id]
            shard_token_count = len(shard_tokens)

        else:
            shard_tokens.extend(tokens)
            shard_token_count += len(tokens)
        
    if shard_token_count > 0:
            shard_save_path = os.path.join(shard_dir, f'shard_{shard_idx:06d}')
            np.save(shard_save_path, shard_tokens)
            
            d_end = time.time()
            dt = d_end - d_start
            print(f'saved to {shard_save_path} with {len(shard_tokens):,} tokens | time to create shard: {dt:.2f} secs')

# %%
create_shards(dataset_iterator)
# %%
import time
time.time()

# %%
