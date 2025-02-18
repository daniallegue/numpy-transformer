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