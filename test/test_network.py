import tensorflow as tf
from matplotlib import pyplot as plt

from src.network.attention import scaled_dot_product_attention
from src.network.masking import create_padding_mask, create_look_ahead_mask, create_mask
from src.network.positional_encoding import positional_encoding


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
    print(create_mask(x))


def test_attention():
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


def print_attention(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)
