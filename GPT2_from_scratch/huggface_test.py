#%%
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import functional as F

# %%
from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

# %%
sd_hf = model_hf.state_dict()
for k,v in sd_hf.items():
    print(k, v.shape)
# %%
sd_hf['transformer.wpe.weight'].view(-1)[:20]

# %%
import matplotlib.pyplot as plt

plt.imshow(sd_hf['transformer.wpe.weight'])

# %%
plt.plot(sd_hf['transformer.wpe.weight'][:,150])
plt.plot(sd_hf['transformer.wpe.weight'][:,200])
plt.plot(sd_hf['transformer.wpe.weight'][:,250])
# %%
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

# %%
generator('The normal distribution is derived by', max_length=400, num_return_sequences=1)
# %%
