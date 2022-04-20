import tensorflow as tf

from src.network.attention import AttentionType
from src.network.transformer import Encoder, Decoder


class BaduraLead(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, h, dff, input_vocab_size, target_vocab_size, rate=0.1):
        super().__init__()

        # Setup Encoder and Decoder
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, h=h, dff=dff,
                               input_vocab_size=input_vocab_size, rate=rate, attention_type=AttentionType.relative)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, h=h, dff=dff,
                               target_vocab_size=target_vocab_size, rate=rate, attention_type=AttentionType.relative)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
