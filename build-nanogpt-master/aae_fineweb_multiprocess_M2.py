#%%
# imports
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

#%%
# Constants
TOKENIZER_NAME = "gpt"
TOKENS_PER_SHARD = 10_000_000
NUM_SHARDS = 100
TOTAL_TOKENS = TOKENS_PER_SHARD * NUM_SHARDS
SHARD_DIR = "token_shards"
remote_name = "sample-10BT"
os.makedirs(SHARD_DIR, exist_ok=True)

#%%
# Initialize the encoder globally in subprocesses
def init_worker():
    global encoder
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

def tokenize_batch(texts):
    global encoder
    return [token for text in texts for token in encoder.encode(text)]

#%%
# Load the dataset with streaming to avoid memory overflow
# Stream the dataset
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name=remote_name, streaming=True)

# Buffering logic
def stream_batches(dataset, batch_size=100):
    batch = []
    for example in dataset:
        batch.append(example['text'])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Main loop
def process_dataset():
    buffer = []
    current_shard = 0
    token_count = 0

    nprocs = max(1, os.cpu_count()//2)
    with Pool(processes=nprocs, initializer=init_worker) as pool:
        for token_lists in tqdm(pool.imap(tokenize_batch, stream_batches(dataset, batch_size=100)), desc="Tokenizing"):
            buffer.extend(token_lists)
            token_count += len(token_lists)

            while len(buffer) >= TOKENS_PER_SHARD:
                shard_tokens = buffer[:TOKENS_PER_SHARD]
                buffer = buffer[TOKENS_PER_SHARD:]

                shard_array = np.array(shard_tokens, dtype=np.uint32)
                shard_path = os.path.join(SHARD_DIR, f"shard_{current_shard:03d}.tok.bin")
                shard_array.tofile(shard_path)

                print(f"✅ Saved shard {current_shard} with {TOKENS_PER_SHARD} tokens")
                current_shard += 1

                if current_shard == NUM_SHARDS:
                    print("✅ Reached target token count.")
                    return
#%%

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # or 'fork' if you're sure it works
    process_dataset()

# %%
tokens = np.fromfile("token_shards/shard_099.tok.bin", dtype=np.uint32)
print(tokens[:-10])
print(len(tokens))
# %%
