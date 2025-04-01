#%%
from collections import Counter
import numpy as np
import copy
import regex as re
# %%
regex = re.compile("""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""") 


# %%
text1 = """ so finally I came up with a GPT2 implementation that's gr123ea45435t."""

# list(bytes(text1, 'utf-8'))
# %%
text1_split = regex.findall(text1)
# print(text1_split)

text1_split_byte_string = [bytes(s, 'utf-8') for s in text1_split]
# print(text1_split_byte_string)

text1_split_ids = [list(bytes(s, 'utf-8')) for s in text1_split]
text1_split_ids


# %%


# %%
text1_split_ids = [list(bytes(s, 'utf-8')) for s in regex.findall(text1)]
print(text1_split_ids)
# %%
pairs = []
for s in text1_split_ids:
    pairs.extend(list(zip(s[:], s[1:])))

pairs

# %%
"""
1 text --> 
2 regex split(list of strings) --> 
3 convert to byte strings (list of byte strings) --> 
4 convert the byte strings to ids (list of lists) --> 
5 create pairs individually from each element in the list of lists from the previous step. Note that the pairings will not cross between elements -->
6 add each pair to a list 

7 get the pair that occurs most often (pair_max) -->
8 generate a new token -->
9 replace the max_pair ids with the new token for each element in the list of lists. this means loop over the list of list, each element is a list of ids. If there are any pairs in that list of ids replace them --> 
10 add the pairing and the new token to the merges dictionary with pair as key and the new token as value
11 go to step 7 and repeat for desired number of merges
"""