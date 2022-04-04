import tensorflow as tf

from src.network.layers import EncoderLayer, DecoderLayer
from src.network.masking import create_padding_mask, create_combined_mask
from src.network.positional_encoding import positional_encoding
from src.settings import SEQUENCE_MAX_LENGTH


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, h, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(SEQUENCE_MAX_LENGTH, d_model)

        self.dec_layers = [DecoderLayer(d_model=d_model, h=h, dff=dff, rate=rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Embedding and Encoding
        x = self.embedding(x)  # Shape: batch_size, target_seq_len, d_model
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # Apply Decoder Layers
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # Shape x: batch_size, target_seq_len, d_model
        return x, attention_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, h, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(SEQUENCE_MAX_LENGTH, self.d_model)

        self.enc_layers = [EncoderLayer(d_model=d_model, h=h, dff=dff, rate=rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Embedding and Encoding
        x = self.embedding(x)  # Shape: batch_size, input_seq_len, d_model
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # Apply Encoder Layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # Shape: batch_size, input_seq_len, d_model


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, h, dff, input_vocab_size, target_vocab_size, rate=0.1):
        super().__init__()

        # Setup Encoder and Decoder
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, h=h, dff=dff,
                               input_vocab_size=input_vocab_size, rate=rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, h=h, dff=dff,
                               target_vocab_size=target_vocab_size, rate=rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        # Create masks
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_combined_mask(tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # Shape: batch_size, inp_seq_len, d_model

        # Shape: batch_size, tar_seq_len, d_model
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # Shape: batch_size, tar_seq_len, target_vocab_size

        return final_output, attention_weights
