from enum import Enum

import tensorflow as tf


class AttentionType(Enum):
    absolute = "absolute"
    relative = "relative"


def scaled_dot_product_attention(q, k, v, mask):
    """ Calculates the attention weights for the queries, keys and values.

    Input consists of queries and keys of dimension `d_k`, values of dimension `d_v`.

    Args:
        q: Queries
        k: Keys
        v: Values
        mask: Mask that will be applied to the product of q and k

    Returns: The calculated attention weights

    """
    # Multiply Q and K^T
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # Shape: (..., seq_len_q, seq_len_k)

    # Scale by root of d_k, in order to optimize softmax
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = matmul_qk / tf.math.sqrt(d_k)

    # Apply mask
    if mask is not None:
        # -1e9 is used as substitute for negative infinity, will make values sufficiently small for softmax
        scaled_qk += (mask * -1e9)

    # Softmax on the last axis

    # Shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_qk, axis=-1)

    # Shape: (..., seq_len_q, d_k)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def relative_scaled_dot_product_attention(q, k, v, e, mask):
    """ Calculates the relative attention weights for the queries, keys and values.

    Input consists of queries and keys of dimension `d_k`, values of dimension `d_v`.

    Args:
        q: Queries
        k: Keys
        v: Values
        e: Position embedding matrix
        mask: Mask that will be applied to the product of q and k

    Returns: The calculated attention weights

    """
    # Shape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # Shape: (..., seq_len_q, seq_len_k)
    s_rel = skew(tf.matmul(q, e, transpose_b=True))

    # Scale by root of d_k, in order to optimize softmax
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = (matmul_qk + s_rel) / tf.math.sqrt(d_k)

    # Apply mask
    if mask is not None:
        # -1e9 is used as substitute for negative infinity, will make values sufficiently small for softmax
        scaled_qk += (mask * -1e9)

    # Softmax on the last axis

    # Shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_qk, axis=-1)

    # Shape: (..., seq_len_q, d_k)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def skew(t: tf.Tensor):
    """ Performs the skewing procedure as introduced by Huang et al.

    Args:
        t: The tensor to skew

    Returns: The skewed tensor

    """
    # Pad input tensor, add no paddings for other than last dimension
    not_padded_dimensions = [[0, 0] for _ in range(len(t.shape) - 1)]
    padded = tf.pad(t, [*not_padded_dimensions, [1, 0]])

    # Reshape
    s_rel = tf.reshape(padded, (-1, padded.shape[-1], padded.shape[-2]))

    # Slice
    s_rel = s_rel[:, 1:]

    return tf.cast(tf.reshape(s_rel, t.shape), t.dtype)
