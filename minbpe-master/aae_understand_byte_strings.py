# %%
"""
after looking at Karpathy's implementation of a decoder, trying to understand behavior of Python bytes method and byte strings.
"""

#the next two lines produce the byte string object of the letter D. Note that bytes takes an iterable as argument
print('D'.encode('utf-8'), type('D'.encode('utf-8')))
print(bytes('D', 'utf-8'), type(bytes('D', 'utf-8'),))
print('-'*40)


# slicing, indexing, concatenating with byte string

t = 'I talked to Fabi'
byte_text = bytes(t, 'utf-8')
print(f'bytes(t, utf-8)  "I talked to Fabi": {byte_text} type:  {type(byte_text)}')

print(f'concatenation of bytes(t, utf-8) of "I talked" with bytes( not very much, utf-8):\n {byte_text[:6] + bytes(' not very much', 'utf-8')}  type: {type(byte_text[:6] + bytes(' not very much', 'utf-8'))}')

print('-'*40)

# convert the byte strings to integer representations. Note that while we can slice or index into this list, we cannot concatenate integer representations. LLMs can only process tokens representes as integers
byte_text_integers = list(byte_text)
print(f"converting the byte string  b'I talked to Fabi' to integer representations:\n{byte_text_integers}  type: {type(byte_text_integers)}")

print('-'*40)

#you can use bytes to convert the integer representations back to byte strings. This is where the byte method has the flexibility to take integer representations or strings and convert them to byte strings. the .encode() method can take only strings. 
print(f'converting an array of integers representing byte strings back to byte strings')
print(bytes(byte_text_integers))



# %%
#converting and array of integers to bytes
integers_to_bytes = [num.to_bytes(2, 'big') for num in byte_text_integers]
print(type(integers_to_bytes[0]))
# nums=[67,68]
# # num_bytes = len(nums)
# bytes_str = [num.to_bytes(2, 'big') for num in nums]
# print(bytes_str)
# print(b''.join(bytes_str).decode('utf-8', errors='replace'))

#%%
byte_text_array = bytearray(t, )

#%%

#encoder:  
# 1) input text --> encode to byte string --> 
# 2) encode to integers --> 
# 3) train new tokens representing pairs (all integers) --> 
# 4) create a merges dictionary with a tuple with the tokens that were paired as keys and an integer representing new tokens (token_ids) starting at 256 as values {(101, 32): 257, (87,12): 258}

#decoder: 
# 1) create a dictionary (Karpathy calls it "vocab") with the token_ids as keys and the byte string representations as values for token_ids 0...255 {162: bytes([162])} ----> 

# 2) add the new tokens created by bpe to vocab by converting the pair from which the token was created to byte string and concatenating {257: vocab[101] + vocab[32]}. Note that we will unmerge by iterating over the merges dictionary in order. Since Python dictionaries now preserve insertion order, the current token we are unpacking will have access to the unpacked version of any token that came before it. So vocab becomes a dictionary with token_ids as keys and byte string representations of the toke_ids as values. For token_ids >255, the values are the byte strings of the concatenated pairs from which the token was created (going back to the root 0..255 token_ids)

# 3) Use the vocab dictionary to convert a list of integer token_ids to byte strings. Concatenate the byte strings in the order that they appear in the token_ids list.

# 4) use decode("utf-8", errors="replace") to convert the byte strings to text

# %%
# %%
