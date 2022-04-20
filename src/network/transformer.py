import tensorflow as tf

from src.network.attention import AttentionType
from src.network.layers import EncoderLayer, DecoderLayer
from src.network.positional_encoding import positional_encoding
from src.settings import SEQUENCE_MAX_LENGTH


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, h, dff, num_encoders, target_vocab_size, rate=0.1,
                 attention_type=AttentionType.standard):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_encoders = num_encoders

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(SEQUENCE_MAX_LENGTH, d_model)

        self.dec_layers = [DecoderLayer(d_model=d_model, h=h, dff=dff, num_encoders=num_encoders, rate=rate,
                                        attention_type=attention_type)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_outputs, training, masks):
        assert len(masks) == self.num_encoders + 1

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Embedding and Encoding
        x = self.embedding(x)  # Shape: batch_size, target_seq_len, d_model
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # Apply Decoder Layers
        for i in range(self.num_layers):
            x, blocks = self.dec_layers[i](x, enc_outputs, training, masks)

            for j, block in enumerate(blocks):
                attention_weights[f'decoder_layer{i + 1}_block{j + 1}'] = block

        # Shape x: batch_size, target_seq_len, d_model
        return x, attention_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, h, dff, input_vocab_size, rate=0.1,
                 attention_type=AttentionType.standard):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(SEQUENCE_MAX_LENGTH, self.d_model)

        self.enc_layers = [EncoderLayer(d_model=d_model, h=h, dff=dff, rate=rate, attention_type=attention_type)
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
    def __init__(self, *, num_layers, d_model, h, dff, num_encoders, input_vocab_sizes, target_vocab_size, rate=0.1,
                 attention_type=AttentionType.standard):
        super().__init__()

        # Setup Encoder and Decoder
        self.encoders = [
            Encoder(num_layers=num_layers, d_model=d_model, h=h, dff=dff, input_vocab_size=input_vocab_sizes[i], rate=rate,
                    attention_type=attention_type) for i in range(num_encoders)]

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, h=h, dff=dff, num_encoders=num_encoders,
                               target_vocab_size=target_vocab_size, rate=rate, attention_type=attention_type)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, enc_masks, dec_masks, training):
        assert len(enc_masks) == len(self.encoders)
        assert len(dec_masks) == len(self.encoders) + 1

        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        # Collect encoder outputs
        enc_outputs = []
        for i, encoder in enumerate(self.encoders):
            enc_outputs.append(encoder(inp[i], training, enc_masks[i]))  # Shape: batch_size, inp_seq_len, d_model

        # Shape: batch_size, tar_seq_len, d_model
        dec_output, attention_weights = self.decoder(tar, enc_outputs, training, dec_masks)

        final_output = self.final_layer(dec_output)  # Shape: batch_size, tar_seq_len, target_vocab_size

        return final_output, attention_weights
