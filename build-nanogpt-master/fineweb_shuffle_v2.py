#%%
"""
My implementation of the Karpathy's nanoGPT code for building a dataset from the fineweb-edu dataset. This implementtion is designed to allow shuffling. Each shard contains complete example documents (Karpathy splits documents between shards). Each document is a tokenized numpy array with the EOT token appended to end. Each shard is a numpy array. Each document is placed in the shard array as a separate numpy object. This will allow the order of the shards as well as the order of the documents within each shard to be shuffled when loading. See aae_dataloader_utils.py
"""

# %%
#imports
#imports
from datasets import load_dataset
import tiktoken
import numpy as np
from tqdm import tqdm
import os
import time
import multiprocessing as mp
import torch

#%%
# Constants
tokenizer_name = "gpt2"
shard_size = 100_000_000 # 100 million tokens per shard
shard_dir = "aae_edu_fineweb10B_shuffle2"
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

# dataset_iterator = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)
# dataset_iterator_test = dataset_iterator.select(range(5))

# # %%
# dataset_iterator_test[0]['text']

#%%
def create_shards(dataset_iterator=None, dataset_iterator_test=None, shard_dir=shard_dir):
    os.makedirs(shard_dir, exist_ok=True)

    print(f'num_worker: {num_workers}')
    with mp.Pool(num_workers) as pool:

        shard_idx = 0
        shard_token_count = 0
        total_token_count = 0
        shard_list = []

        d_start = time.time()
        
        # NOTE: that pool.map outputs a list of  tokens for each example in dataset_iterator  
        for tokens  in tqdm(pool.imap(tokenize, dataset_iterator, chunksize=16), total=len(dataset_iterator), desc=" Percent of datatset processed (cumulative)", unit_scale=True, colour='blue'):
            
            assert len(tokens) <= shard_size, "the length of tokens for a document exceeds the shard size. Please increase the shard size or split the document into smaller chunks."

            if shard_token_count + len(tokens) > shard_size:
                split = 'val' if shard_idx == 0 else 'train'
                shard_save_path = os.path.join(shard_dir, f'{split}_shard_{shard_idx:04d}')

                # Note that we have to make sure that the data type of shard_array_flat and offsets is int and not object. This will allow us to use mmap wehn loading the data. See numpy docs and comments in class DataLoaderShardMultiGPUShuffle2 for more detail on mmap

                # Convert each array to int32 first, then concatenate
                shard_list_int32 = [arr.astype(np.int32) for arr in shard_list]
                
                shard_array_flat = np.concatenate(shard_list_int32).astype(np.int32)
                offsets = np.cumsum([0] + [len(d) for d in shard_list]).astype(np.int64)
                np.savez(shard_save_path, shard_array_flat=shard_array_flat, offsets=offsets)

                # measure time to create this shard
                d_end = time.time()
                dt = d_end - d_start
                d_start = d_end

                print(f'saved to {shard_save_path} with {shard_token_count:,} tokens | time to create shard: {dt:.2f} sec | {shard_token_count / dt:,.0f} tokens/sec  | total tokens: {total_token_count:,}\n')


                # start new shard
                shard_idx += 1
                shard_list= []
                shard_token_count = 0

            else:
                shard_list.append(tokens)
                shard_token_count += len(tokens)
                total_token_count += len(tokens)
            
        if shard_token_count > 0:
                split = 'val' if shard_idx == 0 else 'train'
                shard_save_path = os.path.join(shard_dir, f'{split}_shard_{shard_idx:04d}')
                
                shard_array_flat = np.concatenate(shard_list).astype(np.int32)
                offsets = np.cumsum([0] + [len(d) for d in shard_list]).astype(np.int64)
                np.savez(shard_save_path, shard_array_flat=shard_array_flat, offsets=offsets)
                
                
                d_end = time.time()
                dt = d_end - d_start
                print(f'saved to {shard_save_path} with {shard_token_count:,} tokens | time to create shard: {dt:.2f} sec | {shard_token_count / dt:,.0f} tokens/sec | total tokens: {total_token_count:,}\n')


# %%
if __name__ == '__main__':
    dataset_iterator = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)
    dataset_iterator_test = dataset_iterator.select(range(5))  # Select a subset for testing
    create_shards(dataset_iterator, dataset_iterator_test)




# %%
# test_doc_tokens_list = [np.array([0,1,2,3,4]),
#                         np.array([5,6,7]), 
#                         np.array([8,9,10,11,12,13,14])]

# print(f'test_doc_tokens_list: {len(test_doc_tokens_list)}\n {test_doc_tokens_list}\n')
# test_doc_tokens_flat_array = np.concatenate(test_doc_tokens_list)
# print(f'test_doc_tokens_flat_array:\n {test_doc_tokens_flat_array}\n')

# offsets2 = np.cumsum([0] + [len(d) for d in test_doc_tokens_list])
# print(offsets2)

# num_docs = len(offsets2) - 1
# print(f'\n num_docs: {num_docs}\n')
# shuffled_doc_order = np.random.permutation(num_docs)
# print(f'\n shuffled_doc_order: {shuffled_doc_order}\n')

# shuffled_doc_tokens = [test_doc_tokens_flat_array[offsets2[i] : offsets2[i+1]] for i in shuffled_doc_order]

# print(f'\nshuffled_doc_tokens:\n {shuffled_doc_tokens}')

# print(f"\nshuffled_doc_tokens_tensor:\n{torch.tensor(shuffled_doc_tokens)}")

#%%
data = np.load(f'aae_edu_fineweb10B_shuffle2/train_shard_0001.npz', mmap_mode='r')

print(data['shard_array_flat'][0:10])
print(len(data['offsets']))
# %%
