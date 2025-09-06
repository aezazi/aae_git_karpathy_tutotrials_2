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

#%%
class DataLoaderShardMultiGPUShuffle:
    def __init__(self, B, seq_len , process_rank=0, num_processes=1, split=None, shard_dir='aae_edu_fineweb10B_shuffle'):
            self.B = B
            self.seq_len = seq_len
            self.num_processes = num_processes
            self.process_rank = process_rank
            self.shard_dir = shard_dir
            self.remaining_tokens = []  # to capture the tokens at the end of the current shard that will not be used or seen
            assert split in {'train', 'val'}, f'you must specify if the data is train or val'

            # returns an unordered list of file names in the shard directory. Note that these are just string file names, not the actual numpy arrays.
            self.shard_files = os.listdir(shard_dir) 

            # filter the shard file names to only include those that match the split and sort
            self.shard_files = [file for file in self.shard_files if split in file]
            self.shard_files.sort() 
            

            # this is Karpathy's code. I don't understand why he does this. we already have the shard files in a sorted list and can just add the shard_dir to the file names to get the full path when loading as I have done in the load_tokens_convert_to_tensor method.
            # self.shards = [os.path.join(shard_dir, file) for file in self.shard_files]
            
            # self.reset()  # initialize the current shard index and load the first shard

    def load_tokens_convert_to_tensor(self, shard_file):
        shard_numpy = np.load(f'{self.shard_dir}/{shard_file}')
        return shard_numpy
    
#%%
test = DataLoaderShardMultiGPUShuffle(3,7, split='val')
test.load_tokens_convert_to_tensor(test.shard_files[0])
    

# %%
import numpy as np
array_of_lists = np.load("aae_edu_fineweb10B_shuffle/val_shard_0000.npy", allow_pickle=True)
print(len(array_of_lists[1]))
print(type(array_of_lists[0]))
# %%
from random import shuffle
from glob import glob
files = glob(r"build-nanogpt-master/aae_edu_fineweb10B_shuffle")
shuffle(files)