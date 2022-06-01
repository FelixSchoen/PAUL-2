import numpy as np
import tensorflow as tf

from src.network.attention import scaled_dot_product_attention, AttentionType, relative_scaled_dot_product_attention


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, attention_types, max_relative_distance, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, attention_type=attention_types[0],
                                      max_relative_distance=max_relative_distance)
        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, masks):
        # Multi Head Attention

        # Shape: (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, masks[0])
        attn_output = self.dropout1(attn_output, training=training)
        # Shape: (batch_size, input_seq_len, d_model)
        out1 = self.norm1(x + attn_output)

        # Shape: (batch_size, input_seq_len, d_model)
        pffn_output = self.pffn(out1)
        pffn_output = self.dropout2(pffn_output, training=training)
        # Shape: (batch_size, input_seq_len, d_model)
        out2 = self.norm2(out1 + pffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, num_encoders, attention_types, max_relative_distance, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.num_encoders = num_encoders

        self.mha = [MultiHeadAttention(d_model=d_model, num_heads=num_heads, attention_type=attention_types[i],
                                       max_relative_distance=max_relative_distance)
                    for i in range(num_encoders + 1)]

        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                     for _ in range(num_encoders + 2)]

        self.dropout = [tf.keras.layers.Dropout(rate)
                        for _ in range(num_encoders + 2)]

    def call(self, x, enc_outputs, training, masks):
        assert len(enc_outputs) == self.num_encoders
        assert len(masks) == self.num_encoders + 1

        # enc_ouputs Shapes:(batch_size, input_seq_len, d_model)

        attn_weights = []

        # Shape: (batch_size, target_seq_len, d_model)
        attn, attn_weights_block = self.mha[0](x, x, x, masks[0])
        attn_weights.append(attn_weights_block)
        attn = self.dropout[0](attn, training=training)
        out = self.norm[0](attn + x)

        for enc_ind in range(self.num_encoders):
            # Shape: (batch_size, target_seq_len, d_model)
            attn, attn_weights_block = self.mha[enc_ind + 1](enc_outputs[enc_ind], enc_outputs[enc_ind], out,
                                                             masks[enc_ind + 1])
            attn_weights.append(attn_weights_block)
            attn = self.dropout[enc_ind + 1](attn, training=training)
            # Shape: (batch_size, target_seq_len, d_model)
            out = self.norm[enc_ind + 1](attn + out)

        # Shape: (batch_size, target_seq_len, d_model)
        pffn_output = self.pffn(out)
        pffn_output = self.dropout[-1](pffn_output, training=training)
        # Shape: (batch_size, target_seq_len, d_model)
        out = self.norm[-1](pffn_output + out)

        return out, attn_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, attention_type, max_relative_distance=None):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        assert not (attention_type == AttentionType.self_relative and max_relative_distance is None)

        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.max_len = max_relative_distance

        self.depth = self.d_model // self.num_heads

        self.w_q = tf.keras.layers.Dense(d_model)
        self.w_k = tf.keras.layers.Dense(d_model)
        self.w_v = tf.keras.layers.Dense(d_model)

        if self.max_len is not None:
            self.E = tf.keras.layers.Embedding(self.max_len, self.d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        """ Splits the given tensor into `h` different parts, which can be attended to in parallel.

        Works by splitting the last dimension of the input vector into (num_heads, depth).

        Args:
            x: The tensor to split

        Returns: The representation after the split

        """
        # Split up the last dimension
        t = tf.reshape(x, (*x.shape[:-1], self.num_heads, self.depth))

        # Setup indices for transposition
        last_dimension_index = len(tf.shape(t)) - 1  # Note: Needs len() to work, does not work with rank()
        prior_dimension_indices = np.arange(0, last_dimension_index - 2)

        # Transpose to Shape: (..., num_heads, L, depth)
        return tf.transpose(t, perm=[*prior_dimension_indices, last_dimension_index - 1, last_dimension_index - 2,
                                     last_dimension_index])

    @staticmethod
    def get_embeddings(E, seq_len, max_len):
        """ Builds the position embeddings for relative attention.

        Note that E is considered to be ordered from `-max_len + 1` to `0` on the right. If `seq_len` exceeds `max_len`,
        for the outliers we simply assume a distance of `-max_len +1`.

        Args:
            E: An embedding layer
            seq_len: Length of the current sequence
            max_len: Maximum distances to consider

        Returns: The embeddings

        """
        if not E.built:
            E.build(seq_len)

        if max_len is None:
            # This assumes that E is a Keras embedding layer
            max_len = E.embeddings.get_shape()[0]

        # Sequence is not longer than max_sequence, can fit entirely
        if max_len >= seq_len:
            val = np.arange(-1 * seq_len + 1, 1)
            return E(val)

        # For sequences that are too long, simply set maximum distance for values that are too far apart
        pre = [-1 * max_len + 1 for _ in range(seq_len - max_len)]
        post = np.arange(-1 * max_len + 1, 1)
        val = np.concatenate((pre, post))
        return E(val)

    def call(self, v, k, q, mask):
        # Pipe Q, K, V through the dense layer, adds dimension d_model at the end, shape: (batch_size, seq_len, d_model)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Split Q, K, V for multi-head, adds another dimension, shape: (batch_size, h, seq_len_q, depth)
        batch_size = tf.shape(q)[0]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if self.attention_type == AttentionType.absolute:
            # Apply dot product attention

            # scaled_attention Shape: (batch_size, num_heads, seq_len_q, depth)
            # attention_weights Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        else:
            # Get embedding matrix
            seq_len_k = k.shape[-2]

            # Shape: (seq_len_k, d_model)
            e = MultiHeadAttention.get_embeddings(self.E, seq_len_k, self.max_len)

            # Shape: (seq_len_k, d_model)
            e = self.split_heads(e)

            # rel_scaled_attention Shape: (batch_size, num_heads, seq_len_q, depth)
            # attention_weights Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
            rel_scaled_attention, attention_weights = relative_scaled_dot_product_attention(q, k, v, e, mask=mask)
            scaled_attention = rel_scaled_attention

        # Undo previous transposition
        # Shape: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenate heads
        # Shape: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Shape: (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


class PointwiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, *, d_model, dff):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.layers = tf.keras.Sequential([
            # Shape: (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff, activation="relu"),
            # Shape: (batch_size, seq_len, d_model)
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x):
        return self.layers(x)
