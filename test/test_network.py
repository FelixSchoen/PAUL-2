import tensorflow as tf
from matplotlib import pyplot as plt

from src.network.attention import scaled_dot_product_attention, AttentionType
from src.network.layers import MultiHeadAttention, PointwiseFeedForwardNetwork, EncoderLayer, DecoderLayer
from src.network.masking import create_padding_mask, create_look_ahead_mask
from src.network.positional_encoding import positional_encoding
from src.network.transformer import Encoder, Decoder
from src.util.logging import get_logger

logger = get_logger(__name__)


def test_positional_encoding():
    n, d = 2048, 512
    pos_encoding = positional_encoding(n, d)

    logger.info(f"Shape: {pos_encoding.shape}")
    logger.info(f"Encoding: {pos_encoding}")

    pos_encoding = pos_encoding[0]

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Position')
    plt.xlabel('Depth')
    plt.colorbar()
    plt.show()


def test_padding_mask():
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    logger.info(f"Padding mask: {create_padding_mask(x)}")


def test_look_ahead_mask():
    x = tf.random.uniform((1, 3))
    logger.info(f"Look ahead mask: {create_look_ahead_mask(x.shape[1])}")


def test_scaled_dot_product_attention():
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

    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    logger.info(f"Attention: {temp_attn}")
    logger.info(f"Output: {temp_out}")

    # This query aligns with a repeated key (third and fourth),
    # so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)

    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    logger.info(f"Attention: {temp_attn}")
    logger.info(f"Output: {temp_out}")


def test_multi_head_attention():
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8, attention_type=AttentionType.absolute)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)

    logger.info(f"Output shape: {out.shape}")
    logger.info(f"Attention shape: {attn.shape}")


def test_pointwise_feed_forward_network():
    sample_ffn = PointwiseFeedForwardNetwork(d_model=512, dff=2048)
    shape = sample_ffn(tf.random.uniform((64, 50, 512))).shape

    logger.info(f"Shape: {shape}")


def test_encoder_layer():
    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048, attention_type=AttentionType.absolute)

    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    logger.info(f"Shape: {sample_encoder_layer_output.shape}")

    return sample_encoder_layer_output


def test_decoder_layer():
    sample_encoder_layer_output = test_encoder_layer()

    sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048, num_encoders=1,
                                        attention_type=AttentionType.absolute)

    sample_decoder_layer_output, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)),
                                                          [sample_encoder_layer_output], False, None, [None])

    logger.info(f"Shape: {sample_decoder_layer_output.shape}")


def test_encoder():
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500,
                             attention_type=AttentionType.absolute)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    logger.info(f"Shape: {sample_encoder_output.shape}")

    return sample_encoder_output


def test_decoder():
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000, num_encoders=1, attention_type=AttentionType.absolute)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_outputs=[test_encoder()],
                                  training=False,
                                  self_attention_mask=None,
                                  enc_masks=[None])

    logger.info(f"Output Shape: {output.shape}")
    logger.info(f"Attention Shape: {attn['decoder_layer2_block2'].shape}")
