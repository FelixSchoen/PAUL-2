import numpy as np
from matplotlib import pyplot as plt

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


def test_stuff():
    asdf = np.array([1, 2, 3, 4])
    asdf = asdf[np.newaxis, :]
    print(asdf)
    asdf = asdf[np.newaxis, :]
    print(asdf)
    asdf = asdf[np.newaxis, :]
    print(asdf)
