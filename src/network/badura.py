import json
import os

import tensorflow as tf

from src.util.util import get_project_root


def get_strategy():
    config_file_path = get_project_root() + "/config/tensorflow.json"

    with open(config_file_path, "r") as f:
        os.environ["TF_CONFIG"] = json.load(f)

    # Use NCCL for GPUs
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options)

    return strategy


def train_lead():
    get_strategy()
