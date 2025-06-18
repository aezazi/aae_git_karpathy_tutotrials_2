#%%
"""
My implementation of the Karpathy's nanoGPT code for building a dataset from the fineweb-edu dataset.
"""

# %%
#imports
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
shard_size = 100_000_000 # 100 million tokens per shard
shard_dir = "aae_token_shards_mp"
num_workers = max(1, os.cpu_count() // 2)

#%%
encoder = tiktoken.get_encoding(tokenizer_name)
eot_token_id = encoder.eot_token # get gpt2 tokenizer eot token id
# print(eot_token_id)

# create tokenization function
# NOTE: The use of encoder.encode_ordinary(). This is beacuse When preparing data for training: You usually want full control over where and when special tokens (like <|endoftext|>) are inserted. encode_ordinary gives you just the raw tokenization, so you can insert special tokens like eot_token_id manually, e.g., at shard boundaries or document ends.
def tokenize(example, eot=eot_token_id):
    text = example['text']
    tokens = encoder.encode_ordinary(text)
    tokens.append(eot)
    tokens = np.array(tokens, dtype=np.uint16)
    # print('tokeninzing')
    return tokens


# %%
# NOTE: My approach is different from Karpathy's in that Karpathy splits documents between shards and compeletky fills each shard to the max size.I prefer to keep documents intact and only split them at the end of the document. If a shard is not completely filled, I slice off the portion that is populated so that the final numpy array is exactly shard_size tokens long. This assumes that shard_size is large enough tp accommodate the largest document in the dataset.

def create_shards(dataset_iterator=None, dataset_iterator_test=None, shard_dir=shard_dir):
    os.makedirs(shard_dir, exist_ok=True)

    print(f'num_worker: {num_workers}')
    with mp.Pool(num_workers) as pool:

        shard_idx = 0
        shard_token_count = 0
        shard_list = []

        d_start = time.time()
        
        # NOTE: that pool.map outputs a list of  tokens for each example in dataset_iterator  
        for tokens  in tqdm(pool.imap(tokenize, dataset_iterator, chunksize=16), total=len(dataset_iterator), desc=" Percent of datatset processed (cumulative)", unit_scale=True, colour='blue'):
            
            assert len(tokens) <= shard_size, "the length of tokens for a document exceeds the shard size. Please increase the shard size or split the document into smaller chunks."

            if shard_token_count + len(tokens) > shard_size:
                split = 'val' if shard_idx == 0 else 'train'
                shard_save_path = os.path.join(shard_dir, f'{split}_shard_{shard_idx:04d}')

                # Convert shard_list (list of lists) to a numpy array of list objects
                shard_array = np.array(shard_list, dtype=object) 
                np.save(shard_save_path, shard_array)

                # measure time to create this shard
                d_end = time.time()
                dt = d_end - d_start
                d_start = d_end

                print(f'saved to {shard_save_path} with {shard_token_count:,} tokens | time to create shard: {dt:.2f} sec | {shard_token_count / dt:,.0f} tokens/sec \n')

                # start new shard
                shard_idx += 1
                shard_list= []
                shard_token_count = 0

            else:
                shard_list.append(tokens)
                shard_token_count += len(tokens)
            
        if shard_token_count > 0:
                split = 'val' if shard_idx == 0 else 'train'
                shard_save_path = os.path.join(shard_dir, f'{split}_shard_{shard_idx:04d}')
                shard_array = np.array(shard_list, dtype=object)
                np.save(shard_save_path, shard_array)
                
                d_end = time.time()
                dt = d_end - d_start
                print(f'saved to {shard_save_path} with {shard_token_count:,} tokens | time to create shard: {dt:.2f} sec | {shard_token_count / dt:,.0f} tokens/sec \n')

     
# %%
if __name__ == '__main__':
    dataset_iterator = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)
    dataset_iterator_test = dataset_iterator.select(range(1000))  # Select a subset for testing
    create_shards(dataset_iterator, dataset_iterator_test)

#  %%
# # code to inspect the saved shards
# import numpy as np
# array_of_lists = np.load("aae_token_shards_mp/val_shard_0000.npy", allow_pickle=True)
# array_of_lists[0] = np.concatenate(([eot_token_id], array_of_lists[0]))  # Add eot token at the beginning of the first list
# print(type(array_of_lists))
# print('-' * 40)
# print(array_of_lists[:5])
# print(f'{len(array_of_lists[0]):,}')
# print('-' * 40)

# flat_array = np.fromiter((token for token_list in array_of_lists for token in token_list), dtype=np.uint16)
# print(flat_array[:20])
# print(f'{len(flat_array):,} tokens in the flat array with dtype {flat_array.dtype}')
# %%
