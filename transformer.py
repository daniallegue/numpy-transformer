import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable, Optional


def softmax(x : NDArray, axis : int = -1) -> NDArray:
    """
    :param x: Input array
    :param axis: Axis to compute softmax (Default is last, i.e. -1)
    :return: softmax value
    """

    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum = np.sum(e_x, axis=axis, keepdims=True)

    return e_x / sum

def scaled_dot_product_attention(q : NDArray, k : NDArray, v : NDArray,
    mask : NDArray | None = None) -> Tuple[NDArray, NDArray]:
    """
    :param q: queries matrix
    :param k: keys matrix
    :param v: values matrix
    :param mask: optional mask of shape (..., seq_len_q, seq_len_q) for decoder
    :return: attention mechanism result matrix
    """

    d_k = k.shape[-1]

    score = np.matmul(q, np.swapaxes(k, -1, -2)) # Transpose of K
    score /= np.sqrt(d_k)

    if mask is not None:
        score = np.where(mask, -1e9, score) # Utilise large negative value

    attention_w = softmax(score, axis=1)

    output = np.matmul(attention_w, v)

    return output, attention_w

def multi_head_attention(q : NDArray, k : NDArray, v : NDArray,
    Wq : NDArray, Wk : NDArray, Wv : NDArray, Wo : NDArray,
    mask : NDArray | None = None,
    h : int = 8) -> Tuple[NDArray, NDArray]:
    """
    :param q, k, v: (batch, seq_len, dim)
    :param Wq, Wk, Wv: Linear projection of the matrices
    :param Wo: Linear projection of the output
    :param mask: optional mask of shape (..., seq_len_q, seq_len_q) for decoder
    :param h: attention heads
    :return: [multi-head attention output, attention weights for each head]
    """

    batch_size, seq_len, dim_model = q.shape

    assert dim_model % h == 0, "dim_model must be divisible by h"

    dim_k = dim_model // h

    # Apply linear transformatiions
    proj_q = np.matmul(q, Wq)
    proj_k = np.matmul(k, Wk)
    proj_v = np.matmul(k, Wv)

    # Split into heads
    Q = proj_q.reshape(batch_size, seq_len, h, dim_k)
    K = proj_k.reshape(batch_size, seq_len, h, dim_k)
    V = proj_v.reshape(batch_size, seq_len, h, dim_k)

    # Swap dims to get (batch_size, h, seq_len, dim_k)
    Q = np.swapaxes(Q, 1, 2)
    K = np.swapaxes(K, 1, 2)
    V = np.swapaxes(V, 1, 2)

    if mask is not None and mask.ndim == 4:
        if mask.shape[1] == 1 and mask.shape[2] == seq_len and mask.shape[3] == seq_len:
            mask = np.repeat(mask, h, axis=1)

    # Flatten the matrices
    Q_flatten = Q.reshape(batch_size * h, seq_len, dim_k)
    K_flatten = K.reshape(batch_size * h, seq_len, dim_k)
    V_flatten = V.reshape(batch_size * h, seq_len, dim_k)

    if mask is not None:
        mask_flatten = mask.reshape(batch_size * h, seq_len, seq_len)
    else:
        mask_flatten = None

    attention_out, attention_weights = scaled_dot_product_attention(Q_flatten, K_flatten, V_flatten, mask_flatten)

    attention_out = attention_out.reshape(batch_size, h, seq_len, dim_k)
    attention_weights = attention_weights.reshape(batch_size, h, seq_len, dim_k)

    attention_out = np.swapaxes(attention_out, 1, 2)
    concat_out = attention_out.reshape(batch_size, seq_len, dim_model)

    output = np.matmul(concat_out, Wo)

    return output, attention_weights

def position_wise_ffn(x : NDArray, W1 : NDArray, b1 : NDArray, W2 : NDArray, b2 : NDArray,
                      activation: Callable[[NDArray], NDArray] = None) -> NDArray:
    """
    :param x: Input, size (batch_size, seq_len, dim_model)
    :param W1: (dim_model, dim_ffn)
    :param b1: (dim_ffn, _)
    :param W2: (dim_ffn, dim_model)
    :param b2: (dim_model, _)
    :param activation: A callable activation function which takes a NDArray and returns an NDArray
    :return: transformed output of size (batch_size, seq_len, dim_model)
    """

    if activation is None:
        # ReLU
        def relu(z : NDArray) -> NDArray:
            return np.maximum(0, z)
        activation = relu

    ff1 = np.matmul(x, W1) + b1
    ff1_act = activation(ff1)

    ff2 = np.matmul(ff1_act, W2) + b2

    return ff2

def layer_norm(x : NDArray, gamma: Optional[NDArray] = None, beta : Optional[NDArray] = None,
               eps : float = 1e-6) -> NDArray:
    """
    :param x: (batch_size, seq_len, dim_model)
    :param gamma: (dim_model, _) or None
    :param beta: (dim_model, _) or None
    :param eps: Constant
    :return: Normalzed output
    """

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)

    if gamma is not None:
        x_norm *= gamma # Scale
    if beta is not None:
        x_norm += beta # Shift

    return x_norm

def add_and_norm(x : NDArray, sublayer_out : NDArray,  gamma: Optional[NDArray] = None,
        beta : Optional[NDArray] = None, eps: float = 1e-6) -> NDArray:
    """
    :param x: (batch_size, seq_len, dim_model)
    :param sublayer_out: (batch_size, seq_len, dim_model)
    :param gamma: (dim_model, _) or None
    :param beta: (dim_model, _) or None
    :param eps: Constant
    :return: Normalzed output with sublayer addition
    """

    residual = x + sublayer_out
    out = layer_norm(residual, gamma, beta, eps)

    return out

def positional_encoding(seq_len : int, dim_model : int) -> NDArray:
    """
    :param seq_len: Length of sequence
    :param dim_model: Model dimension
    :return: Positional encoding matrix
    """

    positions = np.arange(seq_len)[:, np.newaxis] # (seq_len, 1)

    dims = np.arange(dim_model)[np.newaxis, :]

    denom = np.power(10000.0, (dims // 2) * 2.0 / dim_model)

    pe = np.zeros((seq_len, dim_model))
    pe[:, 0::2] = np.sin(positions / denom[:, 0::2])
    pe[:, 1::2] = np.cos(positions / denom[:, 1::2])


def encoder_layer(
        x : NDArray,
        Wq: NDArray, Wk: NDArray, Wv : NDArray, Wo : NDArray,
        W1: NDArray, b1 : NDArray, W2 : NDArray, b2 : NDArray,
        mask : NDArray | None = None
) -> NDArray:
    """
    Multi-Head Self Attention + Add & Norm + Feed Forward + Add & Norm
    """

    # Multi Head Attention
    mha_out, _ = multi_head_attention(
        q=x, k=x, v=x,
        Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        mask=mask
    )

    x = add_and_norm(x, mha_out)

    # Feed-Forward Network
    ffn_out = position_wise_ffn(x, W1, b1, W2, b2, activation=np.relu)
    x = add_and_norm(x, ffn_out)

    return x



def decoder_layer(
        x: NDArray,
        enc_out : NDArray,
        Wq_self: np.ndarray, Wk_self: np.ndarray, Wv_self: np.ndarray, Wo_self: np.ndarray,
        Wq_cross: np.ndarray, Wk_cross: np.ndarray, Wv_cross: np.ndarray, Wo_cross: np.ndarray,
        W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
        self_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None
) -> NDArray:
    """
    Masked multi-head attention + add and norm + multi-head cross self attention
    Feed forward + Add & Norm
    """

    # Masked Self Attention (Layer 1)
    mha_out_self, _ = multi_head_attention(
        q=x, k=x, v=x,
        Wq = Wq_self, Wk = Wk_self, Wv = Wv_self, Wo = Wo_self,
        mask = self_mask
    )
    x = add_and_norm(x, mha_out_self)

    # Cross-Attention
    mha_out_cross, _ = multi_head_attention(
        q = x, k = enc_out, v = enc_out,
        Wq=Wq_cross, Wk=Wk_cross, Wv=Wv_cross, Wo=Wo_cross,
        mask=cross_mask
    )
    x = add_and_norm(x, mha_out_cross)

    # FFN
    ffn_out = position_wise_ffn(x, W1, b1, W2, b2, activation=np.relu)
    x = add_and_norm(x, ffn_out)

    return x

def encoder_stack(
        x : NDArray,
        Wq_list, Wk_list, Wv_list, Wo_list,
        W1_list, b1_list, W2_list, b2_list,
        mask: NDArray | None = None,
        N : int = 6
) -> NDArray:
    """
    Stack of N encoder layers
    """

    for i in range(N):
        x = encoder_layer(
            x,
            Wq_list[i], Wk_list[i], Wv_list[i], Wo_list[i],
            W1_list[i], b1_list[i], W2_list[i], b2_list[i],
            mask=mask
        )

    return x


def decoder_stack(
        x: np.ndarray,
        enc_out: np.ndarray,
        Wq_self_list, Wk_self_list, Wv_self_list, Wo_self_list,
        Wq_cross_list, Wk_cross_list, Wv_cross_list, Wo_cross_list,
        W1_list, b1_list, W2_list, b2_list,
        self_mask: np.ndarray | None = None,
        cross_mask: np.ndarray | None = None,
        N: int = 6
) -> NDArray:
    """
    Stack of N decoder layers
    """

    for i in range(N):
        x = decoder_layer(
            x,
            Wq_self_list[i], Wk_self_list[i], Wv_self_list[i], Wo_self_list[i],
            Wq_cross_list[i], Wk_cross_list[i], Wv_cross_list[i], Wo_cross_list[i],
            W1_list[i], b1_list[i], W2_list[i], b2_list[i],
            self_mask = self_mask, cross_mask = cross_mask
        )

    return x

def transformer_forward_pass(
        src_tokens : NDArray,
        target_tokens : NDArray,
        src_embeddings : NDArray,
        target_embeddings : NDArray,
        Wq_enc_list, Wk_enc_list, Wv_enc_list, Wo_enc_list,
        W1_enc_list, b1_enc_list, W2_enc_list, b2_enc_list,
        Wq_self_dec_list, Wk_self_dec_list, Wv_self_dec_list, Wo_self_dec_list,
        Wq_cross_dec_list, Wk_cross_dec_list, Wv_cross_dec_list, Wo_cross_dec_list,
        W1_dec_list, b1_dec_list, W2_dec_list, b2_dec_list,
        final_linear : NDArray,
        src_mask : NDArray | None = None,
        target_mask : NDArray | None = None,
        cross_mask : NDArray | None = None,
        N : int = 6,
        dim_model : int = 512
) -> NDArray:
    """

    :param src_tokens: (batch_size, src_seq_len)
    :param target_tokens: (batch_size, target_seq_len)
    :param src_embeddings: (vocab_size, dim_model)
    :param target_embeddings: (vocab_size, dim_model)
    :param final_linear: (dim_model, vocab_size)
    :return: logits with shape (batch_size, target_seq_len, vocab_size)
    """

    batch_size, src_seq_len = src_tokens.shape
    _, target_seq_len = target_tokens.shape

    src_emb = src_embeddings[src_tokens] # shape (batch_size, src_seq_len, dim_model)

    # Positional encoding
    pe_src = positional_encoding(src_seq_len, dim_model)
    src_emb += pe_src[np.newaxis, ...] # Broadcast whole batch

    # Encoder
    enc_out = encoder_stack(
        src_emb,
        Wq_enc_list, Wk_enc_list, Wv_enc_list, Wo_enc_list,
        W1_enc_list, b1_enc_list, W2_enc_list, b2_enc_list,
        mask=src_mask,
        N=N
    )

    target_emb = target_embeddings[target_tokens]
    pe_target = positional_encoding(target_seq_len, dim_model)
    target_emb += pe_target[np.newaxis, ...]

    # Decoder
    dec_out = decoder_stack(
        target_emb, enc_out,
        Wq_self_dec_list, Wk_self_dec_list, Wv_self_dec_list, Wo_self_dec_list,
        Wq_cross_dec_list, Wk_cross_dec_list, Wv_cross_dec_list, Wo_cross_dec_list,
        W1_dec_list, b1_dec_list, W2_dec_list, b2_dec_list,
        self_mask= target_mask,
        cross_mask=cross_mask,
        N=N
    )

    logits = np.matmul(dec_out, final_linear)

    return logits


