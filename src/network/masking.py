import tensorflow as tf


def create_padding_mask(seq, dim=4):
    """ Creates a padding mask from the given sequence.

    The padding mask contains a `0` where the original sequence had values different from `0`, a `1` where it did not.

    Args:
        seq: The original sequence
        dim: The desired output dimension

    Returns: The mask for the sequence

    """
    # Create mask, has 1 where original had 0, 0 where original has anything else
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Reshape to shape:
    return tf.reshape(mask, (tf.shape(mask)[0], *[1 for _ in range(dim - 2)], tf.shape(mask)[-1]))


def create_look_ahead_mask(size):
    """ Creates a tensor with an upper triangular matrix full of `1`s.

    Args:
        size: Size of the mask

    Returns: The mask

    """
    # Copies the ones, setting everything but the upper triangular matrix to 0
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(mask, tf.float32)


def create_combined_mask(seq, dim=4):
    """ Creates a mask that encompasses both the padding and lookahead mask.

    Args:
        seq: The sequence to create the mask for
        dim: The desired output dimension

    Returns: The combined mask

    """
    padding_mask = create_padding_mask(seq, dim)
    lookahead_mask = create_look_ahead_mask(tf.shape(seq)[-1])

    return tf.maximum(padding_mask, lookahead_mask)
