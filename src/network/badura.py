import tensorflow as tf
from tensorflow.python.data.ops.options import AutoShardPolicy

from src.data_processing.data_pipeline import load_oom_dataset, load_stored_bars, load_dataset_from_bars, \
    load_dataset_from_records
from src.util.logging import get_logger
from src.util.util import get_project_root


def get_strategy():
    config_file_path = get_project_root() + "/config/tensorflow.json"

    # with open(config_file_path, "r") as f:
    #     os.environ["TF_CONFIG"] = f.read().replace("\n", "")

    # Use NCCL for GPUs
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options)

    return strategy


def train_lead():
    logger = get_logger(__name__)

    strategy = get_strategy()

    logger.info("Loading dataset...")
    ds = load_dataset_from_records()

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    ds = ds.with_options(options)

    distributed_ds = strategy.experimental_distribute_dataset(ds)

    logger.info("Counting batches...")
    num_train_batches = len(list(distributed_ds))

    logger.info(f"Counted {num_train_batches} batches.")

    for batch in distributed_ds:
        print(len(batch))

