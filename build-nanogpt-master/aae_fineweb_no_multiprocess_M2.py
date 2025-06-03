#%%
from datasets import load_dataset
import tiktoken
import numpy as np
from tqdm import tqdm
import os

# Constants
TOKENIZER_NAME = "cl100k_base"
TOKENS_PER_SHARD = 10_000_000
NUM_SHARDS = 100
TOTAL_TOKENS = TOKENS_PER_SHARD * NUM_SHARDS
SHARD_DIR = "token_shards"
os.makedirs(SHARD_DIR, exist_ok=True)
remote_name = "sample-10BT"

# Initialize tokenizer
encoder = tiktoken.get_encoding(TOKENIZER_NAME)

# Load the dataset with streaming to avoid memory overflow
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name=remote_name, streaming=True)

# Buffer to accumulate tokens before saving
buffer = []
current_shard = 0
token_count = 0

#%%
# Tokenize and write
for example in tqdm(dataset, desc="Processing dataset"):
    text = example['text']
    tokens = encoder.encode(text)
    buffer.extend(tokens)
    token_count += len(tokens)

    while len(buffer) >= TOKENS_PER_SHARD:
        shard_tokens = buffer[:TOKENS_PER_SHARD]
        buffer = buffer[TOKENS_PER_SHARD:]

        shard_array = np.array(shard_tokens, dtype=np.uint32)
        shard_path = os.path.join(SHARD_DIR, f"shard_{current_shard:03d}.tok.bin")
        shard_array.tofile(shard_path)

        print(f"Saved shard {current_shard} with {TOKENS_PER_SHARD} tokens")
        current_shard += 1

        if current_shard == NUM_SHARDS:
            break

    if current_shard == NUM_SHARDS:
        break

print(f"✅ Done: {current_shard} shards of {TOKENS_PER_SHARD} tokens saved.")

# %%
tokens = np.fromfile("token_shards/shard_099.tok.bin", dtype=np.uint32)
print(tokens[:-10])
print(len(tokens))
# %%
