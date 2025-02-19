```markdown
# Transformer Project README

> **Reference**: This project is based on the architecture and concepts presented in the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al. We implement the core components of the Transformer (Multi-Head Attention, Feed-Forward Networks, Positional Encoding, etc.) in pure JAX, and showcase a training procedure on a dummy dataset using JAX’s `jit` compilation.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Transformer Architecture](#2-transformer-architecture)  
   - [Scaled Dot-Product Attention](#21-scaled-dot-product-attention)  
   - [Multi-Head Attention](#22-multi-head-attention)  
   - [Feed-Forward Network](#23-feed-forward-network)  
   - [Positional Encoding](#24-positional-encoding)  
   - [Add & Norm (Residual Connections)](#25-add--norm-residual-connections)  
3. [Implementation](#3-implementation)  
   - [Functions Created](#functions-created)  
   - [Forward Pass](#forward-pass)  
4. [Training Procedure](#4-training-procedure)  
   - [Parameters (as in the Paper)](#parameters-as-in-the-paper)  
   - [Dummy Dataset and JIT Compilation](#dummy-dataset-and-jit-compilation)  

---

## 1. Overview

The **Transformer** is a sequence-to-sequence model that uses self-attention mechanisms to learn contextual relations between tokens in a sequence. Unlike recurrent or convolutional models, it relies entirely on attention to draw global dependencies between input and output sequences.

Key highlights:

- **No Recurrent or Convolutional Layers**: All context modeling is done through attention mechanisms.  
- **Parallelizable**: Self-attention allows parallel processing of sequences, enabling efficient training.  
- **Positional Encoding**: Injects information about the relative or absolute position of tokens in the sequence.

This project demonstrates how to build and train a simplified Transformer using **JAX** for automatic differentiation and `jit` compilation, referencing the original equations and diagrams from Vaswani et al.

---

## 2. Transformer Architecture

Below is the high-level Transformer architecture from the paper:

![Transformer Architecture](https://raw.githubusercontent.com/tensorflow/tensor2tensor/master/tensor2tensor/visualization/transformer_architecture.png "Transformer Model")

It consists of an **Encoder** stack and a **Decoder** stack. Each layer contains:

1. **Multi-Head Self-Attention** (masked in the decoder’s first sub-layer).  
2. **Add & Norm** (residual connection + layer normalization).  
3. **Feed-Forward Network** (position-wise).  
4. **Add & Norm** again.

### 2.1 Scaled Dot-Product Attention

At the heart of the model is **scaled dot-product attention**. Given query (\(\mathbf{Q}\)), key (\(\mathbf{K}\)), and value (\(\mathbf{V}\)) matrices, the attention output is computed as:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) 
= \text{softmax}\Bigl(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\Bigr) \mathbf{V}
\]

where \(d_k\) is the dimensionality of the keys (and queries).

![Scaled Dot-Product Attention](https://raw.githubusercontent.com/tensorflow/tensor2tensor/master/tensor2tensor/visualization/scaled_dot_product_attention.png "Scaled Dot-Product")

### 2.2 Multi-Head Attention

To allow the model to attend to different positions from different representation subspaces, **multi-head attention** is used. Multiple attention “heads” each compute scaled dot-product attention in parallel, and their outputs are concatenated:

\[
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) 
= \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O
\]

where each head \(i\) is:

\[
\text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q,\, \mathbf{K} \mathbf{W}_i^K,\, \mathbf{V} \mathbf{W}_i^V)
\]

![Multi-Head Attention](https://raw.githubusercontent.com/tensorflow/tensor2tensor/master/tensor2tensor/visualization/multihead_attention.png "Multi-Head Attention")

### 2.3 Feed-Forward Network

After multi-head attention, each position is passed through a **position-wise feed-forward network**:

\[
\text{FFN}(x) = \max(0, x \mathbf{W}_1 + b_1)\,\mathbf{W}_2 + b_2
\]

This is applied identically to each position, separately and identically, hence “position-wise.”

### 2.4 Positional Encoding

Because the model contains no recurrence or convolution, it needs a way to encode sequence order. The **positional encoding** adds sines and cosines of varying frequencies to the embeddings:

\[
\text{PE}_{(pos,\, 2i)} = \sin\Bigl(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\Bigr), \quad
\text{PE}_{(pos,\, 2i+1)} = \cos\Bigl(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\Bigr)
\]

### 2.5 Add & Norm (Residual Connections)

Each sub-layer output is added to the input (residual connection) and then normalized via **Layer Normalization**:

\[
\text{LayerOutput} = \text{LayerNorm}(x + \text{Sublayer}(x))
\]

---

## 3. Implementation

We implement these components in **pure JAX**. Our code structure includes:

- **`scaled_dot_product_attention(q, k, v, mask=None)`**  
- **`multi_head_attention(q, k, v, Wq, Wk, Wv, Wo, mask=None, h=8)`**  
- **`position_wise_ffn(x, W1, b1, W2, b2, activation=...)`**  
- **`layer_norm(x, gamma=None, beta=None, eps=1e-6)`**  
- **`add_and_norm(x, sublayer_out, gamma=None, beta=None, eps=1e-6)`**  
- **`positional_encoding(seq_len, dim_model)`**  
- **`encoder_layer(...)`, `decoder_layer(...)`**  
- **`encoder_stack(...)`, `decoder_stack(...)`**  
- **`transformer_forward_pass(...)`**  

### Functions Created

1. **Scaled Dot-Product Attention**: Implements the equation \(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}\) → softmax → multiply by \(\mathbf{V}\).  
2. **Multi-Head Attention**: Splits queries, keys, values into multiple heads, applies scaled dot-product attention, then concatenates.  
3. **Feed-Forward Network**: Two fully connected layers with a ReLU (or user-defined) activation.  
4. **Positional Encoding**: Returns a matrix of sine/cosine positional encodings.  
5. **Add & Norm**: Implements the residual connection plus layer normalization.  
6. **Encoder/Decoder Layers**: Combines multi-head attention, feed-forward, add & norm.  
7. **Forward Pass**: The full encoder-decoder pass, returning logits.

### Forward Pass

The **`transformer_forward_pass`** function integrates everything:

1. **Embed + Positional Encode** the source tokens.  
2. **Encode** them with `encoder_stack`.  
3. **Embed + Positional Encode** the target tokens.  
4. **Decode** them with `decoder_stack`.  
5. **Final Linear Projection** to obtain logits for each target position.

---

## 4. Training Procedure

We train the model using **JAX** for automatic differentiation and `jit` compilation, which greatly accelerates the forward and backward passes. For demonstration, we use a **dummy dataset** of random token indices.

### Parameters (as in the Paper)

- **Number of Layers** \(N\): 6  
- **Hidden Dimension** \((d_{\text{model}})\): 512  
- **Feed-Forward Dimension** \((d_{\text{ff}})\): 2048  
- **Number of Heads** \((h)\): 8  
- **Vocabulary Size**: 37,000 (example)

### Dummy Dataset and JIT Compilation

1. **Dataset**: We generate random source and target sequences of fixed length (e.g., 50 tokens each).  
2. **Loss Function**: A cross-entropy loss on the predicted logits vs. the true target tokens.  
3. **Autograd and JIT**:  
   - We define `loss_fn` and use `jax.grad` to compute gradients automatically.  
   - We wrap `loss_fn` in `jax.jit` for speed.  
4. **SGD or Adam**: Update parameters in each iteration.  
5. **Print** intermediate losses and track progress.

In practice, you would replace the dummy dataset with real data (e.g., WMT 2014 EN-DE). The code is structured to demonstrate how the components come together in a training loop using JAX.

---
```
