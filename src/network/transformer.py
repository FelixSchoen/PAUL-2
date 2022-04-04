import tensorflow as tf

from src.network.layers import EncoderLayer
from src.network.positional_encoding import positional_encoding
from src.settings import SEQUENCE_MAX_LENGTH


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, h, dff, input_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(SEQUENCE_MAX_LENGTH, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, h=h, dff=dff, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Embedding and encoding
        x = self.embedding(x)  # Shape: batch_size, input_seq_len, d_model
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # Shape: batch_size, input_seq_len, d_model
