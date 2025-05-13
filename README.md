# 🧠 MiniGPT with Custom Tokenizer

A minimal transformer-based language model trained on custom data using a custom-built tokenizer based on Byte Pair Encoding (BPE). This project demonstrates how language models work end-to-end — from tokenization to generation.

---

## 🚀 Project Overview

This project includes:

- ✅ **BPE Tokenizer**: A handcrafted tokenizer that breaks text into subword units using BPE.
- ✅ **Custom Vocabulary Building**: Generates a vocab from corpus and encodes/decodes text.
- ✅ **MiniGPT Model**: A small transformer-based model built using PyTorch.
- ✅ **Training Loop**: Trains the model to predict the next token given a fixed-length context.
- ✅ **Text Generation**: Generates text by autoregressively predicting the next token.

---

## 📦 Folder Structure

```
.
├── Tokenizer.ipynb          # Tokenizer logic (BPE, encoding, decoding)
├── LLM Model.ipynb          # MiniGPT model definition
└── README.md             # You're reading it
```

---

## 🔧 How It Works

### 1. **Tokenizer**
- Implements Byte Pair Encoding (BPE)
- Tokenizes input into subwords based on merge rules
- Converts text ↔ token IDs via dictionaries

```python
def apply_bpe(word, merges): ...
def gpt_tokenizer(corpus, merges): ...
def decode_token_ids(encoded_ids, id2token): ...
```

### 2. **Model Architecture**
```python
class MiniGPT(nn.Module):
    def __init__(...):
        ...
    def forward(x):
        ...
```

- Positional embeddings
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Final projection layer

### 3. **Training Loop**
- Takes context-size `n` inputs (`[t1, t2, ..., tn]`) to predict `t(n+1)`
- Uses cross-entropy loss over vocabulary
- Trained using `Adam` optimizer

```python
for xb, yb in loader:
    logits = model(xb)
    loss = loss_fn(logits[:, -1, :], yb)
```

### 4. **Inference**
- Takes a seed sequence
- Predicts next token one at a time
- Appends predictions and continues

```python
def generate(model, seed_tokens, max_new_tokens): ...
```

---

## 🔢 Requirements

- Python 3.7+
- PyTorch

Install dependencies:

```bash
pip install torch
```

---

## ✍️ Future Improvements

- Add attention masking for proper causal language modeling
- Train on a larger corpus for meaningful outputs
- Use more robust tokenization (like HuggingFace Tokenizers)
- Add batching and padding support

---

## 🤖 Why This Project?

This project helps demystify how LLMs work by rebuilding the core components:

- Tokenization
- Transformer blocks
- Training loop
- Inference engine

If you've ever wanted to build GPT-style models from scratch — this is your playground.

---

## 📝 Author

Built by [Tejas Joshi]

---
