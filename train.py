import jax.numpy as jnp
import jax.random
from jax import Array
from jax import random, grad, jit, tree_util
from typing import Dict, Any, List, Tuple
import numpy as np
from transformer_architecture import (
    transformer_forward_pass,
    embed_tokens,
    positional_encoding
)

JaxArray = Array

def cross_entropy_loss(logits : JaxArray, targets : JaxArray) -> float:
    """
    :param logits: (batch_size, seq_len, vocab_size)
    :param targets: (batch_size, seq_len)
    :return: loss value
    """

    logits_max = jnp.max(logits, axis = -1, keepdims=True)
    exp_logits = jnp.exp(logits - logits_max)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    batch_size, seq_len = targets.shape
    probs_flat = probs.reshape(-1, probs.shape[-1])
    targets_flat = targets.flatten()

    # Negative LL
    loss = -jnp.log(probs_flat[jnp.arange(probs_flat.shape[0]), targets_flat] + 1e-9)
    return jnp.mean(loss)


def init_params(rng : jax.random.PRNGKey,
                dim_model : int = 512,
                vocab_size : int = 37000,
                N : int = 6) -> Dict[str, Any]:
    """
    :param rng: Random Number Generator
    :param dim_model: Model Dimension
    :param vocab_size: Vocabulary Size
    :param N: Number of layers
    :return: Model Params
    """

    params: Dict[str, Any] = {}
    params["src_embeddings"] = random.normal(rng, (vocab_size, dim_model))
    params["target_embeddings"] = random.normal(rng, (vocab_size, dim_model))

    def init_list(shape : Tuple[int, ...]) -> List[JaxArray]:
        return [random.normal(random.fold_in(rng, i), shape) for i in range(N)]


    # Encoder params
    params["Wq_enc_list"] = init_list((dim_model, dim_model))
    params["Wk_enc_list"] = init_list((dim_model, dim_model))
    params["Wv_enc_list"] = init_list((dim_model, dim_model))
    params["Wo_enc_list"] = init_list((dim_model, dim_model))
    params["W1_enc_list"] = init_list((dim_model, 2048))
    params["b1_enc_list"] = init_list((2048,))
    params["W2_enc_list"] = init_list((2048, dim_model))
    params["b2_enc_list"] = init_list((dim_model,))

    # Decoder self-attention params
    params["Wq_self_dec_list"] = init_list((dim_model, dim_model))
    params["Wk_self_dec_list"] = init_list((dim_model, dim_model))
    params["Wv_self_dec_list"] = init_list((dim_model, dim_model))
    params["Wo_self_dec_list"] = init_list((dim_model, dim_model))

    # Decoder cross-attention params
    params["Wq_cross_dec_list"] = init_list((dim_model, dim_model))
    params["Wk_cross_dec_list"] = init_list((dim_model, dim_model))
    params["Wv_cross_dec_list"] = init_list((dim_model, dim_model))
    params["Wo_cross_dec_list"] = init_list((dim_model, dim_model))

    # Decoder ffn
    params["W1_dec_list"] = init_list((dim_model, 2048))
    params["b1_dec_list"] = init_list((2048,))
    params["W2_dec_list"] = init_list((2048, dim_model))
    params["b2_dec_list"] = init_list((dim_model,))

    params["final_linear"] = random.normal(random.fold_in(rng, 9999), (dim_model, vocab_size))

    return params


def loss_fn(params : Dict[str, Any],
            src_tokens : JaxArray,
            target_tokens : JaxArray,
            N : int = 6,
            dim_model : int = 512) -> float:
    """
    :param params: Model params
    :param src_tokens: (batch_size, src_seq_len)
    :param target_tokens: (batch_size, target_seq_len)
    :param N: Number of layers
    :param dim_model: Model dimension
    :return: Loss
    """

    logits = transformer_forward_pass(
        src_tokens=src_tokens,
        target_tokens=target_tokens,
        src_embeddings=params["src_embeddings"],
        target_embeddings=params["target_embeddings"],
        Wq_enc_list=params["Wq_enc_list"],
        Wk_enc_list=params["Wk_enc_list"],
        Wv_enc_list=params["Wv_enc_list"],
        Wo_enc_list=params["Wo_enc_list"],
        W1_enc_list=params["W1_enc_list"],
        b1_enc_list=params["b1_enc_list"],
        W2_enc_list=params["W2_enc_list"],
        b2_enc_list=params["b2_enc_list"],
        Wq_self_dec_list=params["Wq_self_dec_list"],
        Wk_self_dec_list=params["Wk_self_dec_list"],
        Wv_self_dec_list=params["Wv_self_dec_list"],
        Wo_self_dec_list=params["Wo_self_dec_list"],
        Wq_cross_dec_list=params["Wq_cross_dec_list"],
        Wk_cross_dec_list=params["Wk_cross_dec_list"],
        Wv_cross_dec_list=params["Wv_cross_dec_list"],
        Wo_cross_dec_list=params["Wo_cross_dec_list"],
        W1_dec_list=params["W1_dec_list"],
        b1_dec_list=params["b1_dec_list"],
        W2_dec_list=params["W2_dec_list"],
        b2_dec_list=params["b2_dec_list"],
        final_linear=params["final_linear"],
        src_mask=None,
        target_mask=None,
        cross_mask=None,
        N=N,
        dim_model=dim_model
    )

    return cross_entropy_loss(logits, target_tokens)

# JIT-compile for efficiency
loss_fn_jit = jit(loss_fn, static_argnums=(3,4))
grad_fn = jit(grad(loss_fn), static_argnums=(3,4))


def sgd_update(params : Dict[str, Any],
               grads : Dict[str, Any],
               lr : float) -> Dict[str, Any]:
    """
    :param params: Model params
    :param grads: Gradients
    :param lr: Learning rate
    :return: Updated params
    """

    return tree_util.tree_map(lambda p, g : p - lr * g, params, grads)

def train_loop(
        num_epochs : int,
        batch_size : int,
        params : Dict[str, Any],
        dataset : List[Tuple[JaxArray, JaxArray]],
        lr : float,
        N : int = 6,
        dim_model : int = 512
) -> Dict[str, Any]:
    """
    :param num_epochs: Number of epochs
    :param batch_size: Batch size
    :param params: Model params
    :param dataset: List of (src_tokens, target_tokens)
    :param lr: Learning Rate
    :param N: Number of layers
    :param dim_model: Model dimension
    :return: Updated model params
    """

    num_batches = len(dataset) // batch_size
    for epoch in range(num_epochs):
        np.random.shuffle(dataset)
        epoch_loss = 0.0

        for i in range(num_batches):
            batch = dataset[i * batch_size : (i+1) * batch_size]

            src_tokens = jnp.stack([b[0] for b in batch], axis=0)
            target_tokens = jnp.stack([b[1] for b in batch], axis=0)

            loss_val = loss_fn_jit(params, src_tokens, target_tokens, N, dim_model)
            grads = grad_fn(params, src_tokens, target_tokens, N, dim_model)
            params = sgd_update(params, grads, lr)

            epoch_loss += loss_val

            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {i + 1}/{num_batches}, Loss: {loss_val:.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    return params


def main():
    # Hyperparameters following the paper
    num_epochs = 30
    batch_size = 64
    dim_model = 512
    N = 6
    vocab_size = 37000
    src_seq_len = 50
    target_seq_len = 50


    # Dummy dataset
    num_samples = 10000
    dataset : List[Tuple[JaxArray, JaxArray]] = []

    for _ in range(num_samples):
        src = jnp.array(np.random.randint(0, vocab_size, (src_seq_len, )))
        target = jnp.array(np.random.randint(0, vocab_size, (target_seq_len, )))
        dataset.append((src, target))

    rng = random.PRNGKey(0)
    params = init_params(rng, dim_model= dim_model, vocab_size=vocab_size, N = N)

    trained_params = train_loop(num_epochs, batch_size, params, dataset, lr = 1e-4, dim_model=dim_model)

if __name__ == '__main__':
    main()



