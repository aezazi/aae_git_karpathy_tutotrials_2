# class to create data loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import numpy as np
import random

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load the text file for training. Encode the text and convert it to a tensor
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'Batch size: {B}, Sequence length: {T}')
        print(f'Tokens per batch: {(self.B * self.T)}')
        self.batches_per_epoch = len(self.tokens) // (self.B * self.T)

        # this keeps track of where we are in the text for batching
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        self.current_position += B*T # update the current position in the text
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(B, T) # input sequence
        y = buf[1:].view(B, T) # target sequence

        # if loading the next batch would go beyond the end of the training text, reset to the beginning of the text. IT SEEMS TO ME THAT THIS APPROACH LEAVES OUT SOME OF THE LAST TOKENS IN THE TEXT. SO WHATEVER PORTION OF THE TEXT IS NOT USED IN THE LAST BATCH, IT IS NOT USED AT ALL BEACUSE WE RESET TO THE BEGINNING OF THE TEXT.
        if self.current_position + B*T + 1 > len(self.tokens):
            # reset to the beginning of the text
            self.current_position = 0
        
        return x, y
    
class DataLoaderMultiGPU:
    def __init__(self, B, T, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'Batch size: {B}, Sequence length: {T}')
        print(f'Tokens per batch: {(self.B * self.T)}')

        # this keeps track of where we are in the text for batching. Note that the current position is multiplied by the process RANK to ensure that each process gets a different part of the text. The text assigned to each process remains the same throughout training. 
        self.current_position = self.B * self.T * self.process_rank 

    def next_batch(self):
        B, T = self.B, self.T

        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(B, T) # input sequence
        y = buf[1:].view(B, T) # target sequence

        self.current_position += B * T * self.num_processes # update the current position in the text

        # if loading the next batch would go beyond the end of the training text, reset to the beginning of the text
        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            # reset to the beginning of the text that was assisgned to this process.
            self.current_position = self.B * self.T * self.process_rank 
        
        
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        return x, y
    

#%%
#create dataloader for processing shards on multi gpu
class DataLoaderShardMultiGPU:
    def __init__(self, B, seq_len , process_rank=0, num_processes=1, split=None, shard_dir='edu_fineweb10B'):
        self.B = B
        self.seq_len = seq_len
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.shard_dir = shard_dir
        self.remaining_tokens = []  # to capture the tokens at the end of the current shard that will not be used or seen
        assert split in {'train', 'val'}, f'you must specify if the data is train or val'

        # returns an unordered list of file names in the shard directory. Note that these are just string file names, not the actual numpy arrays.
        self.shard_file_names = os.listdir(shard_dir) 

        # filter the shard file names to only include those that match the split and sort
        self.shard_file_names = [name for name in self.shard_file_names if split in name]
        self.shard_file_names.sort() 
        

        # this is Karpathy's code. I don't understand why he does this. we already have the shard files in a sorted list and can just add the shard_dir to the file names to get the full path when loading as I have done in the load_tokens_convert_to_tensor method.
        # self.shards = [os.path.join(shard_dir, file) for file in self.shard_file_names]
        
        # call self.reset() at begining of every training run
        self.reset()  

    
    # Karpathy puts this function outside the class, I think it cleaner to have it as a method of the class. This function loads the tokens from a shard file and converts them to a tensor. 
    
    def reset(self):
        """
        At the very begining of every training run, initialize the current_shard_idx = 0, call
        load_tokens_convert_to_tensor 
        """
        self.current_shard_idx = 0
        self.shard_tensor = self.load_tokens_convert_to_tensor(self.shard_file_names[self.current_shard_idx])
        self.current_position = self.B * self.seq_len * self.process_rank  # set the current position in the text for this process
    
    def load_tokens_convert_to_tensor(self, shard_file):
        
        shard_numpy = np.load(f'{self.shard_dir}/{shard_file}')

        shard_tensor = torch.tensor(shard_numpy, dtype=torch.long)  # convert to tensor with dtype int64

        return shard_tensor
        
    
    def next_batch(self):
        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.shard_tensor[self.current_position:self.current_position + self.B*self.seq_len + 1]
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(self.B, self.seq_len) # input sequence
        y = buf[1:].view(self.B, self.seq_len) # target sequence

        self.current_position += self.B * self.seq_len * self.num_processes # update the current position in the text   

        # if loading the next batch would go beyond the end of the current shard, move to the next shard
        if self.current_position + (self.B*self.seq_len*self.num_processes + 1) > len(self.shard_tensor):
            # really clever code from Karpathy. If currently at the last shard, it produces the index of the next shard file in the list. If currently at last file in the shard file list, it sets the index for the next shard to 0. Very clever  way to loop back to the first file after reaching the end of the shard files list.
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_file_names)

            # capture the tokens at the end of the current shard that will not be used or seen. Note that in Karpathy's implementation, these tokens are just never used.
            remaining_tokens = self.shard_tensor[self.current_position:]
            self.remaining_tokens.append(len(remaining_tokens))
            
            # load the next shard
            self.shard_tensor =  self.load_tokens_convert_to_tensor(self.shard_file_names[self.current_shard_idx])  
            self.current_position = self.B * self.seq_len * self.process_rank  # reset the current position in the text for this process
        
        # move the tensors to the GPU
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        
        return x, y, self.current_shard_idx, self.remaining_tokens
    

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
            
            #shuffle the order of the files in the shard file directory
            random.shuffle(self.shard_file_names)
            
            # initialize the training run
            self.reset()

    def reset(self):
        self.current_shard_idx = 0
        self.shard_tensor = self.load_tokens_convert_to_tensor(new_train_run=True)
        
        self.current_position = self.B * self.seq_len * self.process_rank  # set the current position in the text for this process
    
    def load_tokens_convert_to_tensor(self,  new_train_run = False):
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
    
    def next_batch(self):
        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.shard_tensor[self.current_position:self.current_position + self.B*self.seq_len + 1]
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(self.B, self.seq_len) # input sequence
        y = buf[1:].view(self.B, self.seq_len) # target sequence

        self.current_position += self.B * self.seq_len * self.num_processes # update the current position in the text   

        # if loading the next batch would go beyond the end of the current shard, move to the next shard
        if self.current_position + (self.B*self.seq_len*self.num_processes + 1) > len(self.shard_tensor):
            # really clever code from Karpathy. If currently at the last shard, it produces the index of the next shard file in the list. If currently at last file in the shard file list, it sets the index for the next shard to 0. Very clever  way to loop back to the first file after reaching the end of the shard files list.
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_file_names)

            # capture the tokens at the end of the current shard that will not be used or seen. Note that in Karpathy's implementation, these tokens are just never used.
            remaining_tokens = self.shard_tensor[self.current_position:]
            self.remaining_tokens.append(len(remaining_tokens))
            
            # load the next shard
            self.shard_tensor =  self.load_tokens_convert_to_tensor()  
            self.current_position = self.B * self.seq_len * self.process_rank  # reset the current position in the text for this process
        
        # move the tensors to the GPU
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        
        return x, y, self.current_shard_idx, self.remaining_tokens
    

class DataLoaderShardMultiGPUShuffle2:
    def __init__(self, B, seq_len , process_rank=0, num_processes=1, split=None, shard_dir='aae_edu_fineweb10B_shuffle2', eot=50256):
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
            
            #shuffle the order of the files in the shard file directory
            random.shuffle(self.shard_file_names)
            
            # initialize the training run
            self.reset()

    def reset(self):
        self.current_shard_idx = 0
        self.shard_tensor = self.load_tokens_convert_to_tensor(new_train_run=True)
        
        self.current_position = self.B * self.seq_len * self.process_rank  # set the current position in the text for this process
    
    def load_tokens_convert_to_tensor(self,  new_train_run = False):
         # In this data loader implementation, each shard is is an .npz file which is like .
         # A NumPy .npz file is a compressed archive format used by NumPy to store multiple arrays in a single file. It’s essentially a zip file where each .npy file (which stores a single array) is one member of the archive. Each array inside can be accessed by a key (like a dictionary). In our case,  each .npz file contains 'shard_flat_array' and 'offsets'. shard_flat_array is a flat array of tokens. The tokens are a concatenation of of the tokens for several documents.  Offsets are the begining and ending indices of each document in 'shard_flat_array'. mmap lets you open all shards at once without consuming RAM, then read only the sequences you actually batch into the model. Especially useful if you later train with multiple dataloader workers or distributed GPUs — they can all mmap the same shard file.
        data = np.load(f'{self.shard_dir}/{self.shard_file_names[self.current_shard_idx]}', mmap_mode='r')

        shard_array_flat = data['shard_array_flat']
        offsets = data['offsets']

        # get the number of documents in this shard from the length of the offsets array
        num_docs = len(offsets) - 1

        # np.random.permutation generates an array in range 0 - num_docs in random order. This will be used to access the the offsets start and end indices for documents. Because shuffled_doc_order is in random order, we can use this to produce a shuffling of the docs in this shard
        shuffled_doc_order  = np.random.permutation(num_docs)
        
        # create list of individual doc arrays that are shuffled
        shuffled_docs = [shard_array_flat[offsets[i]:offsets[i+1]] for i in shuffled_doc_order]

        # concatenate back to a single numpy array after shuffling
        self.shard_numpy = np.concatenate(shuffled_docs)

        # if this is the very begining of a training run, insert an eot token at the begining of the first shard array.
        if new_train_run:
             self.shard_numpy = np.concatenate(([self.eot], self.shard_numpy))

        # convert shard_numpy to a tensor with dtype int64
        shard_tensor = torch.tensor(self.shard_numpy, dtype=torch.int64)
        
        return shard_tensor
    
    def next_batch(self):
        # select a sequence of tokens equal to batch size * sequence length + 1 (for the target token)
        buf = self.shard_tensor[self.current_position:self.current_position + self.B*self.seq_len + 1]
        
        # create the input and target sequences from the buffer
        x = buf[:-1].view(self.B, self.seq_len) # input sequence
        y = buf[1:].view(self.B, self.seq_len) # target sequence

        self.current_position += self.B * self.seq_len * self.num_processes # update the current position in the text   

        # if loading the next batch would go beyond the end of the current shard, move to the next shard
        if self.current_position + (self.B*self.seq_len*self.num_processes + 1) > len(self.shard_tensor):
            # really clever code from Karpathy. If currently at the last shard, it produces the index of the next shard file in the list. If currently at last file in the shard file list, it sets the index for the next shard to 0. Very clever  way to loop back to the first file after reaching the end of the shard files list.
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_file_names)

            # capture the tokens at the end of the current shard that will not be used or seen. Note that in Karpathy's implementation, these tokens are just never used.
            remaining_tokens = self.shard_tensor[self.current_position:]
            self.remaining_tokens.append(len(remaining_tokens))
            
            # load the next shard
            self.shard_tensor =  self.load_tokens_convert_to_tensor()  
            self.current_position = self.B * self.seq_len * self.process_rank  # reset the current position in the text for this process
        
        # move the tensors to the GPU
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        
        return x, y, self.current_shard_idx, self.remaining_tokens