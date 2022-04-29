import os
import time
from contextlib import nullcontext
from datetime import datetime
from enum import Enum

import tensorflow as tf
from tensorflow.python.data.ops.options import AutoShardPolicy

from src.network.attention import AttentionType
from src.network.masking import MaskType
from src.network.optimization import TransformerLearningRateSchedule
from src.network.training import Trainer
from src.network.transformer import Transformer
from src.preprocessing.data_pipeline import load_dataset_from_records
from src.settings import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, LEAD_OUTPUT_VOCAB_SIZE, \
    INPUT_VOCAB_SIZE_DIF, PATH_CHECKPOINT_LEAD, BUFFER_SIZE, SHUFFLE_SEED, TRAIN_VAL_SPLIT, SEQUENCE_MAX_LENGTH, EPOCHS, \
    PATH_TENSORBOARD
from src.util.logging import get_logger
from src.util.util import get_src_root


class NetworkType(Enum):
    lead = "lead"
    acmp = "acmp"


def get_strategy():
    config_file_path = get_src_root() + "/config/tensorflow.json"

    with open(config_file_path, "r") as f:
        # os.environ["TF_CONFIG"] = f.read().replace("\n", "")
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Use NCCL for GPUs
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options)

    return strategy


def get_network_objects(network_type, *, strategy, optimizer, train_loss, train_accuracy, val_loss, val_accuracy):
    if network_type == NetworkType.lead:
        transformer = Transformer(num_layers=NUM_LAYERS,
                                  d_model=D_MODEL,
                                  num_heads=NUM_HEADS,
                                  dff=DFF,
                                  input_vocab_sizes=[INPUT_VOCAB_SIZE_DIF],
                                  target_vocab_size=LEAD_OUTPUT_VOCAB_SIZE,
                                  num_encoders=1,
                                  attention_type=AttentionType.relative,
                                  max_relative_distance=SEQUENCE_MAX_LENGTH)
        trainer = Trainer(transformer=transformer, optimizer=optimizer,
                          train_loss=train_loss, train_accuracy=train_accuracy,
                          val_loss=val_loss, val_accuracy=val_accuracy,
                          mask_types=[MaskType.lookahead], strategy=strategy)

        return transformer, trainer
    else:
        raise NotImplementedError


def train_network(network_type, start_epoch=0):
    logger = get_logger(__name__)

    strategy = get_strategy()

    # Create nullcontext if run in single GPU mode
    if strategy is None:
        context = nullcontext()
    else:
        logger.info("Starting in distributed mode.")
        context = strategy.scope()

    with context:
        logical_gpus = tf.config.list_logical_devices("GPU")
        logger.info(f"Running with {logical_gpus} virtual GPUs...")

        logger.info("Loading dataset...")
        ds = load_dataset_from_records()

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        ds = ds.with_options(options)

        amount_batches = len(list(ds.as_numpy_iterator()))
        logger.info(f"Overall dataset consists of {amount_batches} batches.")

        # Split into training and validation data
        logger.info("Splitting into training and validation datasets...")
        train_size = int(TRAIN_VAL_SPLIT * amount_batches)
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)

        # Count batches
        amount_train_batches = len(list(train_ds.as_numpy_iterator()))
        logger.info(f"Train dataset consists of {amount_train_batches} batches.")

        amount_val_batches = len(list(val_ds.as_numpy_iterator()))
        logger.info(f"Validation dataset consists of {amount_val_batches} batches.")

        logger.info("Constructing model...")

        # Load learning rate
        learning_rate = TransformerLearningRateSchedule(D_MODEL)

        # Load optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Loss and Accuracy
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        val_loss = tf.keras.metrics.Mean(name="val_loss")
        val_accuracy = tf.keras.metrics.Mean(name="val_accuracy")

        transformer, trainer = get_network_objects(network_type, strategy=strategy, optimizer=optimizer,
                                                   train_loss=train_loss, train_accuracy=train_accuracy,
                                                   val_loss=val_loss, val_accuracy=val_accuracy)

        # For restarting at a later epoch
        start_epoch = tf.Variable(start_epoch)

        # Set checkpoint
        checkpoint = tf.train.Checkpoint(transformer=transformer,
                                         optimizer=optimizer,
                                         epoch=start_epoch)
        checkpoint_path = PATH_CHECKPOINT_LEAD + "/" + network_type.value
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=10)

        # If checkpoint exists, restore it
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            logger.info(
                f"Restored checkpoint, will start from epoch {start_epoch.numpy()}, {optimizer.iterations.numpy()} "
                f"iterations already completed.")

        # Tensorboard setup
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = PATH_TENSORBOARD + "/" + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        logger.info("Starting training process...")
        for epoch in range(start_epoch.numpy(), EPOCHS):
            # Shuffle dataset
            train_distributed_ds = train_ds \
                .shuffle(BUFFER_SIZE, seed=SHUFFLE_SEED + epoch) \
                .prefetch(tf.data.AUTOTUNE)
            val_distributed_ds = val_ds \
                .prefetch(tf.data.AUTOTUNE)

            if strategy is not None:
                train_distributed_ds = strategy.experimental_distribute_dataset(train_distributed_ds)
                val_distributed_ds = strategy.experimental_distribute_dataset(val_distributed_ds)

            # Reset timers
            epoch_timer = time.time()
            batch_timer = time.time()

            # Reset states
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()

            # Train batches
            for (batch_num, batch) in enumerate(train_distributed_ds):
                # Load data
                lead_seq, lead_dif, acmp_seq, acmp_dif = _load_data(batch)

                # Train step
                if strategy is not None:
                    trainer.distributed_train_step([lead_dif], lead_seq)
                else:
                    trainer.train_step([lead_dif], lead_seq)

                # Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar("train_loss", train_loss.result(), step=optimizer.iterations)
                    tf.summary.scalar("train_accuracy", train_accuracy.result(), step=optimizer.iterations)
                    tf.summary.scalar("val_loss", val_loss.result(), step=optimizer.iterations)
                    tf.summary.scalar("val_accuracy", val_accuracy.result(), step=optimizer.iterations)

                # Logging
                mem_usage = tf.config.experimental.get_memory_info("GPU:0")
                logger.info(
                    f"[E{epoch + 1:02d}B{batch_num + 1:04d}]: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}. "
                    f"Time taken: {round(time.time() - batch_timer, 2):.2f}s ({mem_usage['peak'] / 1e+9 :.2f} GB)")

                # Reset timer
                batch_timer = time.time()
                tf.config.experimental.reset_memory_stats("GPU:0")

            # Validation batches
            for (batch_num, batch) in enumerate(val_distributed_ds):
                # Load data
                lead_seq, lead_dif, acmp_seq, acmp_dif = _load_data(batch)

                # Validation step
                if strategy is not None:
                    trainer.distributed_val_step([lead_dif], lead_seq)
                else:
                    trainer.val_step([lead_dif], lead_seq)

            # Logging
            logger.info(f"[Epoch ended]")
            logger.info(f"[E{epoch + 1:02d}]: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}. "
                        f"Val Loss {val_loss.result():.4f}, Val Accuracy {val_accuracy.result():.4f}. "
                        f"Time taken: {round(time.time() - epoch_timer, 2)}s")

            # Save checkpoint
            checkpoint_save_path = checkpoint_manager.save()
            print(f"[E{epoch + 1:02d}]: Saving checkpoint at {checkpoint_save_path}.")


def _load_data(batch):
    lead_seqs, lead_difs, acmp_seqs, acmp_difs = [], [], [], []

    if hasattr(batch, "values"):
        unstacked = tf.unstack(batch.values[0])
    else:
        unstacked = batch

    for e_lead_seq, e_lead_dif, e_acmp_seq, e_acmp_dif in unstacked:
        lead_seqs.append(e_lead_seq)
        lead_difs.append(e_lead_dif)
        acmp_seqs.append(e_acmp_seq)
        acmp_difs.append(e_acmp_dif)

    lead_seq = tf.stack(lead_seqs)
    lead_dif = tf.stack(lead_difs)
    acmp_seq = tf.stack(acmp_seqs)
    acmp_dif = tf.stack(acmp_difs)

    return lead_seq, lead_dif, acmp_seq, acmp_dif
