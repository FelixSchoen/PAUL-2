import logging

import numpy as np
import tensorflow as tf


def test_mask_multiplication():
    mask = tf.random.uniform(shape=[64, 1, 1, 102])
    qk = tf.random.uniform(shape=[4])

    logging.info(f"Shape mask: {tf.shape(mask)}")
    logging.info(f"Shape qk: {tf.shape(qk)}")

    qk += (mask * -1e9)
    logging.info(f"Shape: {tf.shape(qk)}")


def test_different_dimensions():
    x = tf.constant([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    y = tf.constant([1, 2, 3, 4, 5])

    logging.info(f"Shape x: {tf.shape(x)}")
    logging.info(f"Shape y: {tf.shape(y)}")

    z = y + x

    logging.info(f"Shape z: {tf.shape(z)}")
    logging.info(f"z: {z}")


def test_categorical():
    samples = tf.random.categorical(tf.math.log([[0, 0.5]]), 5)

    print(samples)
    print(tf.math.log([[0, 0.5]]))

    print(0*-np.inf)
