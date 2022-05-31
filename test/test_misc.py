import logging
import os

import tensorflow as tf
from sCoda import Sequence

from src.config.settings import D_TYPE, SEQUENCE_MAX_LENGTH, START_TOKEN, DATA_TRAIN_OUTPUT_FILE_PATH
from src.preprocessing.preprocessing import load_records


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    samples = tf.random.categorical(tf.math.log([[0, 0.5, 1], [1, 2, 3]]), 1)

    print(samples)


def test_tensor_array():
    to_fill = 128

    array = tf.TensorArray(D_TYPE, size=SEQUENCE_MAX_LENGTH, dynamic_size=False)
    for to_f in range(to_fill):
        array = array.write(to_f, 1)
    tensor = array.stack()

    print(tensor)


def test_stack_tensor_arrays():
    size = 5

    arrays = []

    for i in range(size):
        array = tf.TensorArray(D_TYPE, size=SEQUENCE_MAX_LENGTH, dynamic_size=False)
        array = array.write(0, START_TOKEN)
        arrays.append(array)
        print(array.stack())

    array_list = [tensor_array.stack() for tensor_array in arrays]
    print(array_list)
    tensor = tf.stack(array_list)

    print(tensor)


def test_prediction():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    predictions = tf.constant([[1, 2, 3, 5, 4], [6, 15, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=tf.float32)

    am = tf.argmax(predictions, axis=-1)
    print(am)

    ct = tf.squeeze(tf.random.categorical(predictions / 0.1, 1), [-1])
    print(ct)


def test_load_seq():
    lead_seq = Sequence.sequences_from_midi_file("resources/beethoven_027-2_m3_4.mid", [[0]], [])[0]
    print(lead_seq)
    for msg in lead_seq.rel.messages:
        print(msg)


def test_data_shape():
    print(-1e9)
