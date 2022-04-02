import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    """ Calculates the angles used for the positional encoding later on.

    Note that since the input for the positional encoding is `2i` and `2i + 1` respectively, we have to integer
    divide `i` by 2. This function supports matrices as input, as it uses matrix multiplication to apply the formula.

    Args:
        pos: Current position of the element
        i: Current index of the dimension
        d_model: Overall dimension of the model

    Returns: Angles used for positional encoding

    """
    angle = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    return pos * angle


def positional_encoding(position, d_model, dim=3):
    """ Calculates the positional encoding for all combinations of `(position, d_model)`.

    Args:
        position: Maximum position of the model
        d_model: Dimension of the model
        dim: Desired dimension of the output tensor

    Returns: An n-dimensional tensor containing all computed values, `position` in the second-to-last dimension,
    `d_model` in the last.

    """
    # Get angles for entire range of position and d_model
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Replace even values with sine applied to them
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Replace odd values with cosine applied to them
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads

    # Introduce new dimensions
    for _ in range(dim - 2):
        pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
