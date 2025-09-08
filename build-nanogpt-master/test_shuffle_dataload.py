#%%
import os
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import aae_utils
import dataloader_utils
import random

#%%
class DataLoaderShardMultiGPUShuffle:
    def __init__(self, B, seq_len , process_rank=0, num_processes=1, split=None, shard_dir='aae_edu_fineweb10B_shuffle', eot=50256):
            self.B = B
            self.seq_len = seq_len
            self.num_processes = num_processes
            self.process_rank = process_rank
            self.shard_dir = shard_dir
            self.eot = eot
            self.remaining_tokens = []  # to capture the tokens at the end of the current shard that will not be used or seen
            assert split in {'train', 'val'}, f'you must specify if the data is train or val'

            # returns an unordered list of file names in the shard directory. Note that these are just string file names, not the actual numpy arrays.
            self.shard_file_names = os.listdir(shard_dir) 

            # filter the shard file names to only include those that match the split and sort
            self.shard_file_names = [name for name in self.shard_file_names if split in name]
            
            #shuffle the order of the files the shard file directory
            random.shuffle(self.shard_file_names)
            
            

            # this is Karpathy's code. I don't understand why he does this. we already have the shard files in a sorted list and can just add the shard_dir to the file names to get the full path when loading as I have done in the load_tokens_convert_to_tensor method.
            # self.shards = [os.path.join(shard_dir, file) for file in self.shard_file_names]
            
            # initialize the training run
            self.reset()

    
    def reset(self):
        self.current_shard_idx = 0
        self.load_tokens_convert_to_tensor(new_train_run=True)
        
        self.current_position = self.B * self.seq_len * self.process_rank  # set the current position in the text for this process
    
    def load_tokens_convert_to_tensor(self, new_train_run = False):
         # Each shard_file contains  numpy objects each of which is a numpy array of a tokensized document
        shard_file_docs_np_objects = np.load(f'{self.shard_dir}/{self.shard_file_names[self.current_shard_idx]}', allow_pickle=True)

        
        # shuffle shard_file_docs. this shuffles the order of individual documents in this shard
        np.random.shuffle(shard_file_docs_np_objects)

        # unpack the numpy objects (arrays of tokenized documents) into one array
        self.shard_numpy = np.concatenate(shard_file_docs_np_objects)

        # if this is the very begining of a training run, insert an eot token at the begining of the first shard array.
        if new_train_run:
             self.shard_numpy = np.concatenate(([self.eot], self.shard_numpy))

        # convert shard_numpy to a tensor with dtype int64
        shard_tensor = torch.tensor(self.shard_numpy, dtype=torch.int64)
        
        return shard_tensor
    
    
    
#%%
test = DataLoaderShardMultiGPUShuffle(3,7, split='val')
print(test.shard_numpy)

# test.shard_numpy

    

# %%
import numpy as np
shard_dir = "aae_edu_fineweb10B_shuffle"

shard_files = os.listdir(shard_dir)
split = 'val'
shard_files = [file for file in shard_files if split in file]
print(shard_files)
# random.shuffle(shard_files)
# print(f"Contents of aae_edu_fineweb10B_shuffle shuffles:\n{shard_files}")

# %%
shard_file = shard_files[0]
print(f'shard_file: {shard_file}, type: {type(shard_file)}')

shard_file_docs_numpy = np.load(f'{shard_dir}/{shard_files}', allow_pickle=True)
print(f'shard_file_docs_numpy type: {type(shard_file_docs_numpy)}')
print(shard_file_docs_numpy.size)
doc = shard_file_docs_numpy[1]


num_el = 0
for doc in shard_file_docs_numpy:
     s = doc.size
     num_el += s

print(f'num_el: {num_el}')

#%%
unpacked = np.concatenate(shard_file_docs_numpy)
print(unpacked.size == num_el)
print(unpacked[0])
unpacked = np.concatenate(([50256], unpacked))
print(unpacked[0])
print(unpacked.dtype)


# %%
# shard_dir = "aae_edu_fineweb10B_shuffle"
shard_dir = "edu_fineweb10B"

from dataloader_utils import DataLoaderShardMultiGPU
loader = DataLoaderShardMultiGPU(B=3, seq_len=5, split='val')
loader.reset()
print(f'loader.shard_files: {loader.shard_file_names}.  type: {type(loader.shard_file_names)}')


shard_np  = np.load(f'{shard_dir}/{loader.shard_file_names[0]}', allow_pickle=True)
print(f'len shard_np {len(shard_np):,} print shard_np numel: {shard_np.size}')
print(f'{shard_np[1]}')

np.array_equal(shard_np, test.shard_numpy)
     

# count = np.sum(arr == target_value)
# print(shard_np[0:2000])
# %%

