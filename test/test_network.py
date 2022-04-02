import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.network.masking import create_padding_mask, create_look_ahead_mask
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
    print(create_padding_mask(x))

    plt.matshow(create_look_ahead_mask(5))
    print(create_look_ahead_mask(5))
    plt.show()


def test_stuff():
    asdf = np.array([1, 2, 3, 4])
    asdf = asdf[np.newaxis, :]
    print(asdf)
    asdf = asdf[np.newaxis, :]
    print(asdf)
    asdf = asdf[np.newaxis, :]
    print(asdf)
