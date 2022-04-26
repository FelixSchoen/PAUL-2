import os
import time

import tensorflow as tf

from src.data_processing.data_pipeline import load_dataset_from_records
from src.network.attention import AttentionType
from src.network.masking import MaskType
from src.network.optimization import TransformerSchedule
from src.network.training import Trainer
from src.network.transformer import Transformer
from src.settings import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, LEAD_OUTPUT_VOCAB_SIZE, \
    INPUT_VOCAB_SIZE_DIF, PATH_CHECKPOINT_LEAD
from src.util.logging import get_logger
from src.util.util import get_project_root


def get_strategy():
    config_file_path = get_project_root() + "/config/tensorflow.json"

    with open(config_file_path, "r") as f:
        os.environ["TF_CONFIG"] = f.read().replace("\n", "")

    # Use NCCL for GPUs
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options)

    return strategy


def train_lead():
    logger = get_logger(__name__)

    strategy = get_strategy()

    with strategy.scope():
        logger.info("Loading dataset...")
        ds = load_dataset_from_records()

        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        # ds = ds.with_options(options)

        distributed_ds = strategy.experimental_distribute_dataset(ds)

        logger.info("Constructing model...")
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
        train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

        # val_loss = tf.keras.metrics.Mean(name="val_loss")
        # val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

        epochs = 1

        logger.info("Starting training process...")
        for epoch in range(start_epoch.numpy(), epochs):
            start = time.time()
            batch_timer = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch_num, batch) in enumerate(distributed_ds):
                lead_seqs, lead_difs = [], []
                for e_lead_seq, e_lead_dif, _, _ in batch:
                    lead_seqs.append(e_lead_seq)
                    lead_difs.append(e_lead_dif)

                lead_seq = tf.stack(lead_seqs)
                lead_dif = tf.stack(lead_difs)

                trainer = Trainer(strategy, badura_lead, optimizer, train_loss, train_accuracy)

                trainer([lead_dif], lead_seq, [MaskType.padding])

                logger.info(
                    f"[E{epoch+1:03d}B{batch_num:03d}]: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}."
                    f"Time taken: {round(time.time() - batch_timer, 2)}s")

                batch_timer = time.time()

            # if (epoch + 1) % 5 == 0:
            #     ckpt_save_path = ckpt_manager.save()
            #     print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
