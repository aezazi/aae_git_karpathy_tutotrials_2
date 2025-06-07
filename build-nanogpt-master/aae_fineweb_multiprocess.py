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
import multiprocessing as mp

#%%
# Constants
tokenizer_name = "gpt2"
shard_size = 100_000_000
shard_dir = "aae_token_shards_multiprocess"
num_workers = max(1, mp.cpu_count()-1) 
batch_size_multiprocess = 1000

os.makedirs(shard_dir, exist_ok=True)

#%%
encoder = tiktoken.get_encoding(tokenizer_name)
eot_token_id = encoder.eot_token # get gpt2 tokenizer eot token id
print(eot_token_id)
#%%
# Load the dataset with streaming to avoid memory overflow

dataset_iterator = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)

#%%
# create tokenization function
# NOTE: The use of encoder.encode_ordinary(). This is beacuse When preparing data for training: You usually want full control over where and when special tokens (like <|endoftext|>) are inserted. encode_ordinary gives you just the raw tokenization, so you can insert special tokens like eot_token_id manually, e.g., at shard boundaries or document ends.
def tokenize(example: str, eot=eot_token_id):
    text = example['text']
    tokens = encoder.encode_ordinary(text)
    tokens.append(eot)
    print('i am here1')
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

    d_start = time.time()
    with mp.Pool(num_workers) as pool:
        for tokens in pool.map(tokenize, dataset_iterator, chunksize=16):
            p
            # if adding more tokens goes over the shard size limit, save the shard and start a new shard
            if shard_token_count + len(tokens) > shard_size:
                shard_save_path = os.path.join(shard_dir, f'shard_{shard_idx:04d}')
                shard_tokens = np.array(shard_tokens, dtype=np.uint16)
                np.save(shard_save_path, shard_tokens)
                
                # measure time to create this shard
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
                print('concating shard len: {shard_token_count}')

        if shard_token_count > 0:
                shard_save_path = os.path.join(shard_dir, f'shard_{shard_idx:06d}')
                np.save(shard_save_path, shard_tokens)
                
                d_end = time.time()
                dt = d_end - d_start
                print(f'saved to {shard_save_path} with {len(shard_tokens):,} tokens | time to create shard: {dt:.2f} secs')


# %%
if __name__ == "__main__":
    # mp.set_start_method('spawn')
    create_shards(dataset_iterator)

# %%
tokens = np.load("aae_token_shards_multiprocess/shard_000000.npy")
print(tokens[-1])
print(len(tokens))

# %%
a = np.array([1,2,3,4])
b = np.array([7,8,9])
print(np.concatenate((a,b), axis=0))
# %%
