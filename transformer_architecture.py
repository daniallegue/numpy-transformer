import jax.numpy as jnp
from jax import Array
from typing import Tuple, Callable, Optional

JaxArray = Array


def softmax(x: JaxArray, axis: int = -1) -> JaxArray:
    """
    Compute the softmax of an array along a specified axis.

    :param x: Input array.
    :param axis: Axis to compute softmax (Default is last axis).
    :return: Softmax-normalized array.
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    e_x = jnp.exp(x - x_max)
    sum_e = jnp.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e


def scaled_dot_product_attention(q: JaxArray, k: JaxArray, v: JaxArray,
                                 mask: Optional[JaxArray] = None) -> Tuple[JaxArray, JaxArray]:
    """
    Compute scaled dot-product attention.

    :param q: Query matrix.
    :param k: Key matrix.
    :param v: Value matrix.
    :param mask: Optional mask of shape (..., seq_len_q, seq_len_q) for decoder.
    :return: Tuple of (attention output, attention weights).
    """
    d_k = k.shape[-1]
    score = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    score = score / jnp.sqrt(d_k)

    if mask is not None:
        score = jnp.where(mask, -1e9, score)

    attention_w = softmax(score, axis=1)
    output = jnp.matmul(attention_w, v)
    return output, attention_w


def multi_head_attention(q: JaxArray, k: JaxArray, v: JaxArray,
                         Wq: JaxArray, Wk: JaxArray, Wv: JaxArray, Wo: JaxArray,
                         mask: Optional[JaxArray] = None,
                         h: int = 8) -> Tuple[JaxArray, JaxArray]:
    """
    Multi-head attention mechanism.

    :param q, k, v: Arrays of shape (batch, seq_len, dim_model).
    :param Wq, Wk, Wv: Linear projection matrices.
    :param Wo: Output linear projection matrix.
    :param mask: Optional mask of shape (..., seq_len_q, seq_len_q).
    :param h: Number of attention heads.
    :return: Tuple (multi-head attention output, attention weights for each head).
    """
    batch_size, seq_len, dim_model = q.shape
    assert dim_model % h == 0, "dim_model must be divisible by h"
    dim_k = dim_model // h

    # Apply linear transformations.
    proj_q = jnp.matmul(q, Wq)
    proj_k = jnp.matmul(k, Wk)
    proj_v = jnp.matmul(v, Wv)  # Note: usually uses v (not k)

    # Split into heads.
    Q = proj_q.reshape(batch_size, seq_len, h, dim_k)
    K = proj_k.reshape(batch_size, seq_len, h, dim_k)
    V = proj_v.reshape(batch_size, seq_len, h, dim_k)

    # Swap dimensions to (batch_size, h, seq_len, dim_k)
    Q = jnp.swapaxes(Q, 1, 2)
    K = jnp.swapaxes(K, 1, 2)
    V = jnp.swapaxes(V, 1, 2)

    if mask is not None and mask.ndim == 4:
        if mask.shape[1] == 1 and mask.shape[2] == seq_len and mask.shape[3] == seq_len:
            mask = jnp.repeat(mask, h, axis=1)

    # Flatten for attention.
    Q_flatten = Q.reshape(batch_size * h, seq_len, dim_k)
    K_flatten = K.reshape(batch_size * h, seq_len, dim_k)
    V_flatten = V.reshape(batch_size * h, seq_len, dim_k)

    mask_flatten = mask.reshape(batch_size * h, seq_len, seq_len) if mask is not None else None

    attention_out, attention_weights = scaled_dot_product_attention(Q_flatten, K_flatten, V_flatten, mask_flatten)
    attention_out = attention_out.reshape(batch_size, h, seq_len, dim_k)
    # Note: Adjusting attention_weights shape to (batch, h, seq_len, seq_len) might be more standard.
    attention_weights = attention_weights.reshape(batch_size, h, seq_len, seq_len)

    attention_out = jnp.swapaxes(attention_out, 1, 2)
    concat_out = attention_out.reshape(batch_size, seq_len, dim_model)
    output = jnp.matmul(concat_out, Wo)
    return output, attention_weights


def position_wise_ffn(x: JaxArray, W1: JaxArray, b1: JaxArray,
                      W2: JaxArray, b2: JaxArray,
                      activation: Optional[Callable[[JaxArray], JaxArray]] = None) -> JaxArray:
    """
    Position-wise Feed-Forward Network.

    :param x: Input array of shape (batch_size, seq_len, dim_model).
    :param W1: Weight matrix of shape (dim_model, dim_ffn).
    :param b1: Bias vector of shape (dim_ffn,).
    :param W2: Weight matrix of shape (dim_ffn, dim_model).
    :param b2: Bias vector of shape (dim_model,).
    :param activation: Activation function (default: ReLU).
    :return: Transformed output of shape (batch_size, seq_len, dim_model).
    """
    if activation is None:
        activation = lambda z: jnp.maximum(0, z)
    ff1 = jnp.matmul(x, W1) + b1
    ff1_act = activation(ff1)
    ff2 = jnp.matmul(ff1_act, W2) + b2
    return ff2


def layer_norm(x: JaxArray, gamma: Optional[JaxArray] = None,
               beta: Optional[JaxArray] = None, eps: float = 1e-6) -> JaxArray:
    """
    Layer normalization.

    :param x: Input array of shape (batch_size, seq_len, dim_model).
    :param gamma: Scale parameter (optional).
    :param beta: Shift parameter (optional).
    :param eps: Epsilon for numerical stability.
    :return: Normalized output.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta
    return x_norm


def add_and_norm(x: JaxArray, sublayer_out: JaxArray,
                 gamma: Optional[JaxArray] = None, beta: Optional[JaxArray] = None,
                 eps: float = 1e-6) -> JaxArray:
    """
    Apply residual connection followed by layer normalization.

    :param x: Input array (batch_size, seq_len, dim_model).
    :param sublayer_out: Output from a sublayer.
    :param gamma: Scale parameter (optional).
    :param beta: Shift parameter (optional).
    :param eps: Epsilon for numerical stability.
    :return: Normalized output.
    """
    residual = x + sublayer_out
    out = layer_norm(residual, gamma, beta, eps)
    return out


def positional_encoding(seq_len: int, dim_model: int) -> JaxArray:
    """
    Compute sinusoidal positional encodings.

    :param seq_len: Length of sequence.
    :param dim_model: Model dimension.
    :return: Positional encoding matrix of shape (seq_len, dim_model).
    """
    positions = jnp.arange(seq_len)[:, jnp.newaxis]  # (seq_len, 1)
    dims = jnp.arange(dim_model)[jnp.newaxis, :]  # (1, dim_model)
    denom = jnp.power(10000.0, (dims // 2) * 2.0 / dim_model)
    pe = jnp.zeros((seq_len, dim_model))
    pe = pe.at[:, 0::2].set(jnp.sin(positions / denom[:, 0::2]))
    pe = pe.at[:, 1::2].set(jnp.cos(positions / denom[:, 1::2]))
    return pe


def encoder_layer(x: JaxArray,
                  Wq: JaxArray, Wk: JaxArray, Wv: JaxArray, Wo: JaxArray,
                  W1: JaxArray, b1: JaxArray, W2: JaxArray, b2: JaxArray,
                  mask: Optional[JaxArray] = None) -> JaxArray:
    """
    A single encoder layer: Multi-head self-attention + Add & Norm + FFN + Add & Norm.
    """
    mha_out, _ = multi_head_attention(q=x, k=x, v=x,
                                      Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
                                      mask=mask)
    x = add_and_norm(x, mha_out)
    ffn_out = position_wise_ffn(x, W1, b1, W2, b2, activation=jnp.relu)
    x = add_and_norm(x, ffn_out)
    return x


def decoder_layer(x: JaxArray, enc_out: JaxArray,
                  Wq_self: JaxArray, Wk_self: JaxArray, Wv_self: JaxArray, Wo_self: JaxArray,
                  Wq_cross: JaxArray, Wk_cross: JaxArray, Wv_cross: JaxArray, Wo_cross: JaxArray,
                  W1: JaxArray, b1: JaxArray, W2: JaxArray, b2: JaxArray,
                  self_mask: Optional[JaxArray] = None,
                  cross_mask: Optional[JaxArray] = None) -> JaxArray:
    """
    A single decoder layer: Masked self-attention + Add & Norm +
    Cross-attention + Add & Norm + FFN + Add & Norm.
    """
    # Masked self-attention.
    mha_out_self, _ = multi_head_attention(q=x, k=x, v=x,
                                           Wq=Wq_self, Wk=Wk_self, Wv=Wv_self, Wo=Wo_self,
                                           mask=self_mask)
    x = add_and_norm(x, mha_out_self)
    # Cross-attention.
    mha_out_cross, _ = multi_head_attention(q=x, k=enc_out, v=enc_out,
                                            Wq=Wq_cross, Wk=Wk_cross, Wv=Wv_cross, Wo=Wo_cross,
                                            mask=cross_mask)
    x = add_and_norm(x, mha_out_cross)
    # Feed-forward.
    ffn_out = position_wise_ffn(x, W1, b1, W2, b2, activation=jnp.relu)
    x = add_and_norm(x, ffn_out)
    return x


def encoder_stack(x: JaxArray,
                  Wq_list, Wk_list, Wv_list, Wo_list,
                  W1_list, b1_list, W2_list, b2_list,
                  mask: Optional[JaxArray] = None,
                  N: int = 6) -> JaxArray:
    """
    Stack N encoder layers.
    """
    for i in range(N):
        x = encoder_layer(x,
                          Wq_list[i], Wk_list[i], Wv_list[i], Wo_list[i],
                          W1_list[i], b1_list[i], W2_list[i], b2_list[i],
                          mask=mask)
    return x


def decoder_stack(x: JaxArray, enc_out: JaxArray,
                  Wq_self_list, Wk_self_list, Wv_self_list, Wo_self_list,
                  Wq_cross_list, Wk_cross_list, Wv_cross_list, Wo_cross_list,
                  W1_list, b1_list, W2_list, b2_list,
                  self_mask: Optional[JaxArray] = None,
                  cross_mask: Optional[JaxArray] = None,
                  N: int = 6) -> JaxArray:
    """
    Stack N decoder layers.
    """
    for i in range(N):
        x = decoder_layer(x, enc_out,
                          Wq_self_list[i], Wk_self_list[i], Wv_self_list[i], Wo_self_list[i],
                          Wq_cross_list[i], Wk_cross_list[i], Wv_cross_list[i], Wo_cross_list[i],
                          W1_list[i], b1_list[i], W2_list[i], b2_list[i],
                          self_mask=self_mask, cross_mask=cross_mask)
    return x


def transformer_forward_pass(src_tokens: JaxArray,
                             target_tokens: JaxArray,
                             src_embeddings: JaxArray,
                             target_embeddings: JaxArray,
                             Wq_enc_list, Wk_enc_list, Wv_enc_list, Wo_enc_list,
                             W1_enc_list, b1_enc_list, W2_enc_list, b2_enc_list,
                             Wq_self_dec_list, Wk_self_dec_list, Wv_self_dec_list, Wo_self_dec_list,
                             Wq_cross_dec_list, Wk_cross_dec_list, Wv_cross_dec_list, Wo_cross_dec_list,
                             W1_dec_list, b1_dec_list, W2_dec_list, b2_dec_list,
                             final_linear: JaxArray,
                             src_mask: Optional[JaxArray] = None,
                             target_mask: Optional[JaxArray] = None,
                             cross_mask: Optional[JaxArray] = None,
                             N: int = 6,
                             dim_model: int = 512) -> JaxArray:
    """
    The full Transformer forward pass.

    :param src_tokens: (batch_size, src_seq_len)
    :param target_tokens: (batch_size, target_seq_len)
    :param src_embeddings: (vocab_size, dim_model)
    :param target_embeddings: (vocab_size, dim_model)
    :param final_linear: (dim_model, vocab_size)
    :return: Logits of shape (batch_size, target_seq_len, vocab_size)
    """
    batch_size, src_seq_len = src_tokens.shape
    _, target_seq_len = target_tokens.shape

    src_emb = embed_tokens(src_tokens, src_embeddings)  # (batch_size, src_seq_len, dim_model)
    pe_src = positional_encoding(src_seq_len, dim_model)
    src_emb = src_emb + pe_src[jnp.newaxis, ...]  # Broadcast over batch

    # Encoder.
    enc_out = encoder_stack(src_emb,
                            Wq_enc_list, Wk_enc_list, Wv_enc_list, Wo_enc_list,
                            W1_enc_list, b1_enc_list, W2_enc_list, b2_enc_list,
                            mask=src_mask, N=N)

    target_emb = embed_tokens(target_tokens, target_embeddings)
    pe_target = positional_encoding(target_seq_len, dim_model)
    target_emb = target_emb + pe_target[jnp.newaxis, ...]

    # Decoder.
    dec_out = decoder_stack(target_emb, enc_out,
                            Wq_self_dec_list, Wk_self_dec_list, Wv_self_dec_list, Wo_self_dec_list,
                            Wq_cross_dec_list, Wk_cross_dec_list, Wv_cross_dec_list, Wo_cross_dec_list,
                            W1_dec_list, b1_dec_list, W2_dec_list, b2_dec_list,
                            self_mask=target_mask, cross_mask=cross_mask, N=N)

    logits = jnp.matmul(dec_out, final_linear)
    return logits


def embed_tokens(tokens: JaxArray, embedding_matrix: JaxArray) -> JaxArray:
    """
    Embed token indices using the embedding matrix.

    :param tokens: (batch_size, seq_len) token indices.
    :param embedding_matrix: (vocab_size, dim_model) embedding lookup table.
    :return: Embedded tokens of shape (batch_size, seq_len, dim_model).
    """
    return embedding_matrix[tokens]
