import json
import os

import tensorflow as tf
from tensorflow.core.framework.dataset_options_pb2 import AutoShardPolicy

from src.data_processing.data_pipeline import load_oom_dataset
from src.util.util import get_project_root


def get_strategy():
    config_file_path = get_project_root() + "/config/tensorflow.json"

    with open(config_file_path, "r") as f:
        print(json.load(f))
        # os.environ["TF_CONFIG"] =

    # Use NCCL for GPUs
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options)

    return strategy


def train_lead():
    strategy = get_strategy()
    ds = load_oom_dataset()

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    ds.with_options(options)

    distributed_ds = strategy.experimental_distribute_dataset(ds)

    num_train_batches = len(list(distributed_ds))

    print(f"There are {num_train_batches} train batches")
