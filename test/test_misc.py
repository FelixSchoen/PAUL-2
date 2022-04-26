import time

import tensorflow as tf

from src.util.logging import get_logger
from src.util.util import get_project_root


def test_logger():
    LOGGER = get_logger("root." + __name__)
    LOGGER.info("Test")
    time.sleep(2)


def test_root_path():
    print(get_project_root())


def test_tensor():
    tensor = tf.random.uniform(shape=[3, 3, 3], dtype=tf.int32, maxval=9)
    unstacked_tensor = tf.unstack(tensor)

    print(tensor)
    print(len(unstacked_tensor))
