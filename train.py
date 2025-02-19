import jax.numpy as jnp
from jax import Array
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



