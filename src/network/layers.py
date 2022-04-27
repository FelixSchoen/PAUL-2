import tensorflow as tf

from src.network.attention import scaled_dot_product_attention, AttentionType


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, attention_type, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, attention_type=attention_type)
        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Multi Head Attention

        # Shape: (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
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
    def __init__(self, *, d_model, num_heads, dff, num_encoders, attention_type, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.num_encoders = num_encoders

        self.mha = [MultiHeadAttention(d_model=d_model, num_heads=num_heads, attention_type=attention_type)
                    for _ in range(num_encoders + 1)]

        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                     for _ in range(num_encoders + 2)]

        self.dropout = [tf.keras.layers.Dropout(rate)
                        for _ in range(num_encoders + 2)]

    def call(self, x, enc_outputs, training, self_attention_mask, enc_masks):
        assert len(enc_outputs) == self.num_encoders
        assert len(enc_masks) == self.num_encoders

        # enc_ouputs Shapes:(batch_size, input_seq_len, d_model)

        attn_weights = []

        # Shape: (batch_size, target_seq_len, d_model)
        attn, attn_weights_block = self.mha[0](x, x, x, self_attention_mask)
        attn_weights.append(attn_weights_block)
        attn = self.dropout[0](attn, training=training)
        out = self.norm[0](attn + x)

        for enc_ind in range(self.num_encoders):
            # Shape: (batch_size, target_seq_len, d_model)
            attn, attn_weights_block = self.mha[0](enc_outputs[enc_ind], enc_outputs[enc_ind], out, enc_masks[enc_ind])
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
    def __init__(self, *, d_model, num_heads, attention_type):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_type = attention_type

        self.depth = self.d_model // self.num_heads

        self.w_q = tf.keras.layers.Dense(d_model)
        self.w_k = tf.keras.layers.Dense(d_model)
        self.w_v = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """ Splits the given tensor into `h` different parts, which can be attended to in parallel.

        Args:
            x: The tensor to split
            batch_size: Size of the batch

        Returns: The representation after the split

        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # TODO x = tf.reshape(x, (*x.shape[:-1], self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        # Pipe Q, K, V through the dense layer, adds dimension d_model at the end, shape: (batch_size, seq_len, d_model)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Split Q, K, V for multi-head, adds another dimension (splits last into (h, depth))
        batch_size = tf.shape(q)[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if self.attention_type == AttentionType.absolute:
            # Apply dot product attention

            # scaled_attention Shape: (batch_size, num_heads, seq_len_q, depth)
            # attention_weights Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        else:
            raise NotImplementedError

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
