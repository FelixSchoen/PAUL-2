import numpy as np
import tensorflow as tf

from src.network.attention import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, *, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % self.h == 0

        self.depth = d_model // self.h

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wo = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        """ Splits the given tensor into `h` different parts, which can be attended to in parallel.

        Args:
            x: The tensor to split

        Returns: The representation after the split

        """
        # Split last dimension into (h, depth), in order to fit into multiple heads
        x = tf.reshape(x, (*x.shape[:-1], self.h, self.depth))

        # Setup indices for transposition
        last_dimension_index = tf.rank(x) - 1
        prior_dimension_indices = np.arange(0, last_dimension_index + 1)

        # Transpose to Shape: ..., h, L, depth
        return tf.transpose(x, perm=[*prior_dimension_indices[:-3], last_dimension_index - 1, last_dimension_index - 2,
                                     last_dimension_index])

    def call(self, v, k, q, mask):
        # Pipe Q, K, V through the dense layer, adds dimension d_model at the end
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split Q, K, V for multi-head, adds another dimension (splits last into (h, depth))
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        batch_size = tf.shape(q)[0]

        # Apply dot product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # Undo previous transposition
        last_dimension_index = tf.rank(scaled_attention) - 1
        prior_dimension_indices = np.arange(0, last_dimension_index + 1)
        scaled_attention = tf.transpose(scaled_attention, perm=[*prior_dimension_indices[:-3],
                                                                last_dimension_index - 1,
                                                                last_dimension_index - 2,
                                                                last_dimension_index])

        # Concatenate heads
        concat_attention = tf.reshape(scaled_attention, (*scaled_attention.shape[:-2], self.d_model))

        # Apply dense layer
        output = self.wo(concat_attention)  # Shape: batch_size, seq_len_q, d_model

        return output, attention_weights


class PointwiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, *, d_model, dff):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.layers = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x):
        return self.layers(x)
