import tensorflow as tf
from matplotlib import pyplot as plt

from src.network.attention import scaled_dot_product_attention
from src.network.layers import MultiHeadAttention, PointwiseFeedForwardNetwork, EncoderLayer, DecoderLayer
from src.network.masking import create_padding_mask, create_look_ahead_mask, create_combined_mask
from src.network.optimization import TransformerSchedule
from src.network.positional_encoding import positional_encoding
from src.network.transformer import Decoder, Encoder, Transformer
from src.settings import D_MODEL


def test_positional_encoding():
    n, d = 2048, 512
    pos_encoding = positional_encoding(n, d)
    print(pos_encoding.shape)
    print(pos_encoding)
    pos_encoding = pos_encoding[0]

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Position')
    plt.xlabel('Depth')
    plt.colorbar()
    plt.show()


def test_padding_mask():
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    create_padding_mask(x)
    create_look_ahead_mask(5)
    print(create_combined_mask(x))


def test_dot_product_attention():
    print()
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_attention(temp_q, temp_k, temp_v)


def test_multi_head_attention():
    temp_mha = MultiHeadAttention(d_model=512, h=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print((out.shape, attn.shape))


def test_pointwise_feed_forward_network():
    sample_ffn = PointwiseFeedForwardNetwork(d_model=512, dff=2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)


def test_encoder_layer():
    sample_encoder_layer = EncoderLayer(d_model=512, h=8, dff=2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    print(sample_encoder_layer_output.shape)


def test_decoder_layer():
    sample_encoder_layer = EncoderLayer(d_model=512, h=8, dff=2048)

    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    sample_decoder_layer = DecoderLayer(d_model=512, h=8, dff=2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)),
                                                             sample_encoder_layer_output, False, None, None)

    print(sample_decoder_layer_output.shape)


def test_decoder():
    sample_encoder = Encoder(num_layers=2, d_model=512, h=8,
                             dff=2048, input_vocab_size=8500)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

    sample_decoder = Decoder(num_layers=2, d_model=512, h=8,
                             dff=2048, target_vocab_size=8000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)


def test_transformer():
    sample_transformer = Transformer(
        num_layers=2, d_model=512, h=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer([temp_input, temp_target], training=False)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)


def test_learning_schedule():
    temp_learning_rate_schedule = TransformerSchedule(D_MODEL)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel('Learning Rate')
    plt.xlabel('Train Step')
    plt.show()


def print_attention(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)
