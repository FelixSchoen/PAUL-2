import tensorflow as tf
from tensorflow.python.data.ops.options import AutoShardPolicy

from src.data_processing.data_pipeline import load_dataset_from_records
from src.network.attention import AttentionType
from src.network.optimization import TransformerSchedule
from src.network.transformer import Transformer
from src.settings import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, LEAD_OUTPUT_VOCAB_SIZE, \
    INPUT_VOCAB_SIZE_DIF, PATH_CHECKPOINT_LEAD
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

    with strategy.scope():
        badura_lead = Transformer(
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            h=NUM_HEADS,
            dff=DFF,
            num_encoders=1,
            input_vocab_sizes=[INPUT_VOCAB_SIZE_DIF],
            target_vocab_size=LEAD_OUTPUT_VOCAB_SIZE,
            attention_type=AttentionType.relative
        )

        # Load learning rate
        learning_rate = TransformerSchedule(D_MODEL)

        # Load optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

        # For restarting at a later epoch
        start_epoch = tf.Variable(0)

        # TODO: Checkpoint if main worker
        # Set checkpoint
        checkpoint = tf.train.Checkpoint(transformer=badura_lead,
                                         optimizer=optimizer,
                                         epoch=start_epoch)

        checkpoint_path = PATH_CHECKPOINT_LEAD

        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

        # If checkpoint exists, restore it
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            logger.info(
                f"Restored checkpoint, will start from epoch {start_epoch.numpy()}, {optimizer.iterations.numpy()} "
                f"iterations already completed.")

        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        val_loss = tf.keras.metrics.Mean(name="val_loss")
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
