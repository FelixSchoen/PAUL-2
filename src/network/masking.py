from enum import Enum

import tensorflow as tf


class MaskType(Enum):
    padding = "padding"
    lookahead = "lookahead"
    singleout = "singleout"


def create_padding_mask(seq):
    """ Creates a padding mask from the given sequence.

    The padding mask contains a `0` where the original sequence had values different from `0`, a `1` where it did not.

    Args:
        seq: The original sequence

    Returns: The mask for the sequence

    """
    # Create mask, has 1 where original had 0, 0 where original has anything else
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Reshape to shape: batch_size, 1, 1, seq_len
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """ Creates a tensor with an upper triangular matrix full of `1`s.

    Args:
        size: Size of the mask

    Returns: The mask

    """
    # Copies the ones, setting everything but the upper triangular matrix to 0
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(mask, tf.float32)


def create_single_out_mask(size):
    """ Creates a tensor with an upper and lower triangular matrix full of `1`s.

    Args:
        size: Size of the mask

    Returns: The mask

    """
    # Copies the ones, setting the diagonal to 0
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), 0, 0)
    return tf.cast(mask, tf.float32)


def create_combined_mask(seq, mask_fn=create_look_ahead_mask):
    """ Creates a mask that encompasses both the padding and lookahead mask.

    Args:
        seq: The sequence to create the mask for
        mask_fn: A function yielding a mask

    Returns: The combined mask

    """
    padding_mask = create_padding_mask(seq)
    lookahead_mask = mask_fn(tf.shape(seq)[-1])

    return tf.maximum(padding_mask, lookahead_mask)
