#%%
"""

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
shard_size = 100_000_000
shard_dir = "aae_token_shards_mp"
num_workers = max(1, os.cpu_count()-1)

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


#%%
# # test tokenize function
# test = dataset_iterator[0]['text']
# print(test)
# test_tokenize = tokenize(dataset_iterator[0])
# print(test_tokenize)
# print(len(test_tokenize))
# print(len(tokenize(dataset_iterator[1])))
# print(len(tokenize(dataset_iterator[2])))

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

def create_shards(dataset_iterator=None, dataset_iterator_test=None, shard_dir=shard_dir):
    os.makedirs(shard_dir, exist_ok=True)

    print(f'num_worker: {num_workers}')
    with mp.Pool(num_workers) as pool:

        shard_idx = 0
        # initialize shard tokens with eot token at the begining to signal that this document is a continuation
        shard_tokens_buffer = np.empty(shard_size, dtype=np.uint16)
        shard_tokens_buffer[0] = eot_token_id
        # print(f'shard buffer len: {len(shard_tokens_buffer)}')
        shard_token_count = 1

        d_start = time.time()
        
        # NOTE: that pool.map outputs a list of  tokens for each example in dataset_iterator 
        # creating the 'results' variable is how we can wrap the pool output in tqdm for showing progress bars. wraping pool.imap directly in tqdm did not work.
        results = tqdm(pool.imap(tokenize, dataset_iterator), total=len(dataset_iterator), desc=" Percent of datatset processed (cumulative)", unit_scale=True, colour='blue')
        
        for tokens  in results:
            if shard_token_count + len(tokens) > shard_size:
                split = 'val' if shard_idx == 0 else 'train'
                shard_save_path = os.path.join(shard_dir, f'{split}_shard_{shard_idx:04d}')
                shard_tokens_final = shard_tokens_buffer[:shard_token_count].astype(np.uint16)
                np.save(shard_save_path, shard_tokens_final)

                # measure time to create this shard
                d_end = time.time()
                dt = d_end - d_start
                d_start = d_end

                print(f'saved to {shard_save_path} with {len(shard_tokens_final):,} tokens | time to create shard: {dt:.2f} sec | {len(shard_tokens_final) / dt:,.0f} tokens/sec \n')

                # start new shard
                shard_idx += 1
                shard_tokens_buffer = np.empty(shard_size)
                shard_tokens_buffer[0] = eot_token_id
                shard_token_count = 1
            else:
                shard_tokens_buffer[shard_token_count : shard_token_count+len(tokens)] = tokens
                shard_token_count += len(tokens)
            
        if shard_token_count > 0:
                shard_save_path = os.path.join(shard_dir, f'shard_{shard_idx:04d}')
                shard_tokens_final = shard_tokens_buffer[:shard_token_count].astype(np.uint16)
                np.save(shard_save_path, shard_tokens_final)
                
                d_end = time.time()
                dt = d_end - d_start
                print(f'saved to {shard_save_path} with {len(shard_tokens_final):,} tokens | time to create shard: {dt:.2f} secs')

     
# %%
if __name__ == '__main__':
    dataset_iterator = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False)
    dataset_iterator_test = dataset_iterator.select(range(1000000))
    create_shards(dataset_iterator, dataset_iterator_test)

# %%
# tokens = np.load("aae_token_shards_multiprocess_M2/shard_0000.npy")
# print(type(tokens))
# print(tokens[0])
# print(f'{len(tokens):,}')
# %%
