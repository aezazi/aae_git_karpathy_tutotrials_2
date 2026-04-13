# Learning Journey: Building a GPT-2 Style Transformer from Scratch

This repository documents my personal learning journey through several of [Andrej Karpathy's](https://github.com/karpathy) tutorials on building large language models from the ground up.

**This is not production code.** It is a record of exploration and learning. The inline comments throughout are much closer to class notes — capturing questions, observations, and "aha" moments — rather than anything resembling professional documentation.

---

## Tutorials Covered

| Tutorial | Karpathy Source | Local Directory |
|---|---|---|
| micrograd — building an autograd engine | [micrograd](https://github.com/karpathy/micrograd) | `micrograd-master/` |
| minbpe — building a BPE tokenizer | [minbpe](https://github.com/karpathy/minbpe) | `minbpe-master/` |
| GPT-2 building blocks | [makemore / nanoGPT lectures](https://www.youtube.com/watch?v=kCc8FmEb1nY) | `GPT2_building_blocks_from_scratch/` |
| Full GPT-2 + advanced training | [build-nanogpt](https://github.com/karpathy/build-nanogpt) | `build-nanogpt-master/` |

---

## Repository Structure

```
.
├── micrograd-master/          # Scalar autograd engine (~150 lines) — backprop fundamentals
├── minbpe-master/             # Byte Pair Encoding tokenizer — how LLMs tokenize text
├── GPT2_building_blocks_from_scratch/   # Step-by-step GPT building blocks
└── build-nanogpt-master/      # Full GPT-2 style model with advanced training
```

---

## Learning Progression

### 1. `micrograd-master/` — Understanding Back-propagation

A tiny (~100 line) scalar-valued autograd engine implementing reverse-mode autodiff over a dynamically built computation graph, paired with a small neural network library (~50 lines) with a PyTorch-like API.

**Key files:**
- [demo.ipynb](micrograd-master/demo.ipynb) — trains a 2-layer MLP binary classifier on a moon dataset
- [trace_graph.ipynb](micrograd-master/trace_graph.ipynb) — Graphviz visualizations of computation graphs, showing data and gradient flow

---

### 2. `minbpe-master/` — Understanding Tokenization

A minimal Byte Pair Encoding (BPE) tokenizer — the tokenization algorithm used in GPT-2/3/4.

**Key files:**
- [aae_exercise_BasicTokenizer.py](minbpe-master/aae_excercise_BasicTokenizer.py) — my exercise implementation of `BasicTokenizer` (pair counting, merge generation, encode/decode)
- [aae_exercise_RegexTokenizer.py](minbpe-master/aae_excercise_RegexTokenizer.py) — regex-based variant that splits text before BPE (GPT-2 style)
- [aae_tokenization_follow_tutorial_1.py](minbpe-master/aae_tokenization_follow_tutorial_1.py) — step-by-step tutorial follow-along
- [aae_understand_byte_strings.py](minbpe-master/aae_understand_byte_strings.py) — exploration of UTF-8 byte handling
- [Tokenization.ipynb](minbpe-master/Tokenization.ipynb) — notebook with tokenization examples

---

### 3. `GPT2_building_blocks_from_scratch/` — GPT Fundamentals

Incremental implementations building up the components of a transformer, starting from a simple character-level bigram model and adding complexity step by step.

**Key files (in learning order):**
- [aae_gpt_bigram.py](GPT2_buiding_blocks_from_scratch/aae_gpt_bigram.py) — character-level bigram model on tiny-shakespeare; establishes the encode/decode/train loop
- [aae_create_mask_attn_example.py](GPT2_buiding_blocks_from_scratch/aae_create_mask_attn_example.py) — educational demo of causal masking using `torch.tril`
- [aae_gpt_attention.py](GPT2_buiding_blocks_from_scratch/aae_gpt_attention.py) — adds multi-head self-attention to the bigram model
- [aae_sinusoidal_pos_encoding.py](GPT2_buiding_blocks_from_scratch/aae_sinusoidal_pos_encoding.py) — exploration of sinusoidal positional encoding (sin/cos at even/odd dimensions)
- [aae_gpt_attn_sin_pos_temp.py](GPT2_buiding_blocks_from_scratch/aae_gpt_attn_sin_pos_temp.py) — combines attention with sinusoidal positional encoding
- [gpt_dev.py](GPT2_buiding_blocks_from_scratch/gpt_dev.py) — full implementation converted from Karpathy's Colab notebook; complete training loop on tiny-shakespeare

---

### 4. `build-nanogpt-master/` — Full GPT-2 and Beyond

A full GPT-2 style implementation extended with more advanced techniques including Rotary Positional Embeddings (RoPE), Mixture of Experts (MoE), and distributed training with FSDP.

**Model files:**
- [model_base.py](build-nanogpt-master/model_base.py) — base GPT-2 model (`GPTConfig`: block_size=1024, vocab_size=50304, 12 layers, 12 heads, 768 embed dim). Implements `CausalSelfAttention`, `MLP`, `TransformerBlock`.
- [model_rotary.py](build-nanogpt-master/model_rotary.py) — RoPE variant; replaces sinusoidal encoding with rotation matrices applied to Q/K vectors
- [model_moe_fsdp.py](build-nanogpt-master/model_moe_fsdp.py) — **final best-performing model.** Mixture of Experts with FSDP (Fully Sharded Data Parallel); implements `TopKMoEGate` router with load balancing
- [model_moe_fsdp_parallel.py](build-nanogpt-master/model_moe_fsdp_parallel.py) — an attempt to manually shard expert parameters across GPUs (see note below)

**A note on distributed training experiments:** FSDP shards optimizer state and gradients, but still loads all model parameters on every GPU. To address this I experimented with DeepSpeed, but was unable to get it working. I then tried manually parallelizing expert parameters across GPUs in `model_moe_fsdp_parallel.py`. The model trained, but was excruciatingly slow — clearly a flawed implementation. `model_moe_fsdp.py` and `train_moe_fsdp.py` represent the final, best-performing approach.

**Data pipeline:**
- [fineweb.py](build-nanogpt-master/fineweb.py) — Karpathy's original FineWeb-Edu dataset downloader; tokenizes with GPT-2 tokenizer and saves to shards (100M tokens/shard)
- [aae_fineweb.py](build-nanogpt-master/aae_fineweb.py) — my customized fork with additional logging
- [fineweb_shuffle_v2.py](build-nanogpt-master/fineweb_shuffle_v2.py) — document-level shuffling variant; stores documents as separate objects within shards

**Training:**
- [train_moe_fsdp.py](build-nanogpt-master/train_moe_fsdp.py) — **final training script** for the best-performing MoE+FSDP model; distributed process group init, gradient accumulation, LR scheduling, HellaSwag evaluation
- [terminal_output_1.3B_full_run.txt](build-nanogpt-master/terminal_output_1.3B_full_run.txt) — captured terminal output from a complete 1.3B parameter training run using `model_moe_fsdp.py` + `train_moe_fsdp.py`

**Utilities & evaluation:**
- [hellaswag.py](build-nanogpt-master/hellaswag.py) — HellaSwag commonsense reasoning benchmark; compares against GPT-2 and GPT-2-XL baselines
- [aae_utils.py](build-nanogpt-master/aae_utils.py) — `ConfigureOptimizer` helper: groups parameters for weight decay (excludes bias/layernorm), sets up fused AdamW for GPU

**Training artifacts:**
- `hella_accuracy/` — HellaSwag accuracy logs from runs across different model variations
- `train_loss/` — training loss logs from runs across different model variations

---

## A Note on Code Style

The code is intentionally over-commented. Rather than clean, minimal documentation, the comments reflect the learning process — questions I had, things that surprised me, comparisons to concepts I already knew. If you're learning the same material, these notes might be useful. If you're looking for clean reference implementations, Karpathy's original repos are the better source.
