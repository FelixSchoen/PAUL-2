import tensorflow as tf

from src.config.settings import SEQUENCE_MAX_LENGTH, D_TYPE
from src.exception.exceptions import UnexpectedValueException
from src.network.layers import EncoderLayer, DecoderLayer
from src.network.masking import create_padding_mask, create_combined_mask, MaskType, create_single_out_mask, \
    create_look_ahead_mask
from src.network.positional_encoding import positional_encoding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, attention_type, max_relative_distance,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(SEQUENCE_MAX_LENGTH, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, attention_type=attention_type,
                         max_relative_distance=max_relative_distance, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Embedding and Encoding

        # Inp Shape: (batch_size, input_seq_len)
        # Out Shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # Apply Dropout
        x = self.dropout(x, training=training)

        # Apply Encoder Layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # Shape: (batch_size, input_seq_len, d_model)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, target_vocab_size, num_encoders, attention_type,
                 max_relative_distance, rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_encoders = num_encoders

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(SEQUENCE_MAX_LENGTH, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, num_encoders=num_encoders,
                         attention_type=attention_type, max_relative_distance=max_relative_distance, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_outputs, training, self_attention_mask, dec_masks):
        assert len(enc_outputs) == self.num_encoders
        assert len(dec_masks) == self.num_encoders

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Embedding and Encoding

        # Shape: (batch_size, target_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # Apply Dropout
        x = self.dropout(x, training=training)

        # Apply Decoder Layers
        for i in range(self.num_layers):
            x, blocks = self.dec_layers[i](x, enc_outputs, training, self_attention_mask, dec_masks)

            for j, block in enumerate(blocks):
                attention_weights[f'decoder_layer{i + 1}_block{j + 1}'] = block

        # Shape: (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_sizes, target_vocab_size, num_encoders,
                 mask_types_enc, mask_types_dec, attention_type, max_relative_distance, rate=0.1):
        super().__init__()

        self.num_encoders = num_encoders
        self.mask_types_enc = mask_types_enc
        self.mask_types_dec = mask_types_dec

        assert num_encoders == len(input_vocab_sizes)
        assert num_encoders == len(mask_types_enc)
        assert num_encoders == len(mask_types_dec)

        self.encoders = [
            Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                    input_vocab_size=input_vocab_sizes[i], attention_type=attention_type,
                    max_relative_distance=max_relative_distance, rate=rate)
            for i in range(num_encoders)
        ]

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                               target_vocab_size=target_vocab_size, num_encoders=num_encoders,
                               attention_type=attention_type, max_relative_distance=max_relative_distance, rate=rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, omni_input, training):
        # Pass all inputs in first argument
        inputs, target = omni_input

        # Create encoder masks
        enc_masks = []
        for inp, mask_type in zip(inputs, self.mask_types_dec):
            enc_masks.append(Transformer._build_mask(mask_type, inp))

        # Create decoder masks
        dec_self_attention_mask = create_combined_mask(target)
        dec_masks = []
        for inp, mask_type in zip(inputs, self.mask_types_dec):
            dec_masks.append(Transformer._build_mask(mask_type, inp, shift_by=1))

        # Collect Encoder Outputs
        enc_outputs = []
        for i, encoder in enumerate(self.encoders):
            # Shape: (batch_size, inp_seq_len, d_model)
            enc_outputs.append(encoder(inputs[i], training, enc_masks[i]))

        # Shape: (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(target, enc_outputs, training, dec_self_attention_mask, dec_masks)

        # Shape: (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def build_model(self):
        def _get_uniform():
            return tf.cast(tf.random.uniform((1, SEQUENCE_MAX_LENGTH), minval=0, maxval=9, dtype=tf.int32),
                           dtype=D_TYPE)

        inputs = [_get_uniform() for _ in range(self.num_encoders)]
        target = _get_uniform()[:, :-1]

        _, _ = self([inputs, target], False)

    @staticmethod
    def _build_mask(mask_type, inp, shift_by=0):
        if mask_type == MaskType.padding:
            mask = create_padding_mask(inp)
        elif mask_type == MaskType.lookahead:
            mask = create_combined_mask(inp, mask_fn=create_look_ahead_mask)
            mask = mask[:, :, shift_by:, :]
        elif mask_type == MaskType.singleout:
            mask = create_combined_mask(inp, mask_fn=create_single_out_mask)
            mask = mask[:, :, shift_by:, :]
        else:
            raise UnexpectedValueException("Unknown masking type")

        return mask
