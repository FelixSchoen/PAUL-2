import numpy as np
import tensorflow as tf

from src.exception.exceptions import UnexpectedValueException
from src.network.attention import scaled_dot_product_attention, relative_scaled_dot_product_attention, AttentionType
from src.settings import SEQUENCE_MAX_LENGTH


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, h, dff, rate=0.1, attention_type=AttentionType.standard):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)
        self.mha2 = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)

        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # Encoder Output Shape: batch_size, input_seq_len, d_model

        # Calculate self attention
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(attn1 + x)  # Shape: batch_size, input_seq_len, d_model

        # Calculate attention over encoder output
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(attn2 + out1)  # Shape: batch_size, input_seq_len, d_model

        # Pointwise Feed Forward
        pffn_output = self.pffn(out2)
        pffn_output = self.dropout3(pffn_output, training=training)
        out3 = self.norm3(pffn_output + out2)  # Shape: batch_size, input_seq_len, d_model

        return out3, attn_weights_block1, attn_weights_block2


class DecoderLeadLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, h, dff, rate=0.1, attention_type=AttentionType.standard):
        super(DecoderLeadLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)
        self.mha2 = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)

        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, masks):
        # Encoder Output Shape: batch_size, input_seq_len, d_model

        # Calculate self attention
        attn1, attn_weights_block1 = self.mha1(x, x, x, masks[0])
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(attn1 + x)  # Shape: batch_size, input_seq_len, d_model

        # Calculate attention over encoder output
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, masks[1])
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(attn2 + out1)  # Shape: batch_size, input_seq_len, d_model

        # Pointwise Feed Forward
        pffn_output = self.pffn(out2)
        pffn_output = self.dropout3(pffn_output, training=training)
        out3 = self.norm3(pffn_output + out2)  # Shape: batch_size, input_seq_len, d_model

        return out3, [attn_weights_block1, attn_weights_block2]


class DecoderAcmpLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, h, dff, rate=0.1, attention_type=AttentionType.standard):
        super(DecoderAcmpLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)
        self.mha2 = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)
        self.mha3 = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)

        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_outputs, training, masks):
        # Encoder Output Shape: batch_size, input_seq_len, d_model

        # Calculate self attention
        attn1, attn_weights_block1 = self.mha1(x, x, x, masks[0])
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(attn1 + x)  # Shape: batch_size, input_seq_len, d_model

        # Calculate attention over first encoder output
        attn2, attn_weights_block2 = self.mha2(enc_outputs[0], enc_outputs[0], out1, masks[1])
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(attn2 + out1)  # Shape: batch_size, input_seq_len, d_model

        # Calculate attention over second encoder output
        attn3, attn_weights_block3 = self.mha2(enc_outputs[1], enc_outputs[1], out2, masks[2])
        attn3 = self.dropout3(attn3, training=training)
        out3 = self.norm3(attn3 + out2)  # Shape: batch_size, input_seq_len, d_model

        # Pointwise Feed Forward
        pffn_output = self.pffn(out3)
        pffn_output = self.dropout4(pffn_output, training=training)
        out4 = self.norm4(pffn_output + out3)  # Shape: batch_size, input_seq_len, d_model

        return out4, [attn_weights_block1, attn_weights_block2, attn_weights_block3]


class DecoderMultiLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, h, dff, amount_encoders, rate=0.1, attention_type=AttentionType.standard):
        super(DecoderMultiLayer, self).__init__()

        self.mha = [MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type) for _ in
                    range(amount_encoders + 1)]

        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(amount_encoders + 2)]

        self.dropout = [tf.keras.layers.Dropout(rate) for _ in range(amount_encoders + 2)]

    def call(self, x, enc_outputs, training, masks):
        # Encoder Output Shape: batch_size, input_seq_len, d_model

        attn_weights = []

        # Calculate self attention
        attn, attn_weights_block = self.mha[0](x, x, x, masks[0])
        attn_weights.append(attn_weights_block)
        attn = self.dropout[0](attn, training=training)
        out = self.norm[0](attn + x)

        # Calculate attention over encoder outputs
        for enc_ind in range(len(enc_outputs)):
            attn, attn_weights_block = self.mha[enc_ind + 1](enc_outputs[enc_ind], enc_outputs[enc_ind], out,
                                                             masks[enc_ind + 1])
            attn_weights.append(attn_weights_block)
            attn = self.dropout[enc_ind + 1](attn, training=training)
            out = self.norm[enc_ind + 1](attn + out)

        # Pointwise Feed Forward
        pffn_output = self.pffn(out)
        pffn_output = self.dropout[-1](pffn_output, training=training)
        out = self.norm[-1](pffn_output + out)  # Shape: batch_size, input_seq_len, d_model

        return out, attn_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, h, dff, rate=0.1, attention_type=AttentionType.standard):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model=d_model, h=h, use_bias=False, attention_type=attention_type)
        self.pffn = PointwiseFeedForwardNetwork(d_model=d_model, dff=dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Multi Head Attention
        attn_output, _ = self.mha(x, x, x, mask)  # Shape: batch_size, input_seq_len, d_model
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)  # Shape: batch_size, input_seq_len, d_model

        # Pointwise Feed Forward
        pffn_output = self.pffn(out1)  # Shape: batch_size, input_seq_len, d_model
        pffn_output = self.dropout2(pffn_output, training=training)
        out2 = self.norm2(out1 + pffn_output)  # Shape: batch_size, input_seq_len, d_model

        return out2


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, *, d_model, h, use_bias=False, attention_type=AttentionType.standard,
                 max_rel_dist=SEQUENCE_MAX_LENGTH):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.attention_type = attention_type
        self.max_len = max_rel_dist

        assert d_model % self.h == 0

        self.depth = d_model // self.h

        self.wq = tf.keras.layers.Dense(d_model, use_bias=use_bias)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=use_bias)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=use_bias)
        self.wo = tf.keras.layers.Dense(d_model, use_bias=use_bias)

        # For relative position embedding
        self.E = tf.keras.layers.Embedding(self.max_len, self.d_model)

    def split_heads(self, x):
        """ Splits the given tensor into `h` different parts, which can be attended to in parallel.

        Args:
            x: The tensor to split

        Returns: The representation after the split

        """
        # Split last dimension into (h, depth), in order to fit into multiple heads
        x = tf.reshape(x, (*x.shape[:-1], self.h, self.depth))

        # Setup indices for transposition
        last_dimension_index = len(x.shape) - 1  # Note: Needs len() to work, does not work with shape()
        prior_dimension_indices = np.arange(0, last_dimension_index + 1)

        # Transpose to Shape: ..., h, L, depth
        return tf.transpose(x, perm=[*prior_dimension_indices[:-3], last_dimension_index - 1, last_dimension_index - 2,
                                     last_dimension_index])

    @staticmethod
    def get_position_embeddings(E, seq_len, max_len=None):
        """ Builds the position embeddings for relative attention.

        Args:
            E: An embedding layer
            seq_len: Length of the current sequence
            max_len: Maximum distances to consider

        Returns: The embeddings

        """
        if not E.built:
            E.build(seq_len)

        if max_len is None:
            max_len = E.embeddings.get_shape()[0]

        # Sequence is not longer than max_sequence, can fit entirely
        if max_len >= seq_len:
            return E(np.arange(max_len - seq_len, max_len))

        # For sequences that are too long, simply assume maximum distance
        return tf.concat(
            values=[*[E(np.arange(0, 1)) for _ in range(seq_len - max_len)],
                    E(np.arange(0, max_len))],
            axis=0
        )

    def call(self, v, k, q, mask):
        # Pipe Q, K, V through the dense layer, adds dimension d_model at the end
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split Q, K, V for multi-head, adds another dimension (splits last into (h, depth))
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if self.attention_type == "standard":
            # Apply dot product attention
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        elif self.attention_type == "relative":
            seq_len_k = k.shape[-2]
            e = self.get_position_embeddings(self.E, seq_len_k, self.max_len)  # Shape: (seq_len_k, d_model)
            e = self.split_heads(e)
            scaled_attention, attention_weights = relative_scaled_dot_product_attention(q, k, v, e, mask=mask)
        else:
            raise UnexpectedValueException("Unknown attention type")

        # Undo previous transposition
        last_dimension_index = len(
            scaled_attention.shape) - 1  # Note: Needs len() to work, does not work with shape() TODO: Try rank
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
