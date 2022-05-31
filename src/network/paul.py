import os
import time
from contextlib import nullcontext
from datetime import datetime
from operator import itemgetter

import tensorflow as tf
from sCoda import Sequence, Message
from sCoda.util.util import get_note_durations, get_tuplet_durations
from tensorflow.python.data.ops.options import AutoShardPolicy

from src.config.settings import LEAD_OUTPUT_VOCAB_SIZE, \
    INPUT_VOCAB_SIZE_DIF, PATH_CHECKPOINT, BUFFER_SIZE, SHUFFLE_SEED, SEQUENCE_MAX_LENGTH, EPOCHS, \
    PATH_TENSORBOARD, ACMP_OUTPUT_VOCAB_SIZE, INPUT_VOCAB_SIZE_MLD, PATH_SAVED_MODEL, DATA_TRAIN_OUTPUT_FILE_PATH, \
    DATA_VAL_OUTPUT_FILE_PATH, SETTINGS_LEAD_TRANSFORMER, SETTINGS_ACMP_TRANSFORMER, DIFFICULTY_VALUE_SCALE, PATH_MIDI
from src.network.attention import AttentionType
from src.network.generator import Generator, TemperatureSchedule
from src.network.masking import MaskType
from src.network.optimization import TransformerLearningRateSchedule
from src.network.training import Trainer
from src.network.transformer import Transformer
from src.preprocessing.preprocessing import load_records
from src.util.enumerations import NetworkType
from src.util.logging import get_logger
from src.util.util import get_src_root, convert_difficulty, get_prj_root


def get_strategy():
    config_file_path = get_src_root() + "/config/tensorflow.json"

    with open(config_file_path, "r") as f:
        os.environ["TF_CONFIG"] = f.read().replace("\n", "")
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Use NCCL for GPUs
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

    # Get multi worker strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options)

    return strategy


def get_network_objects(network_type, *, strategy=None, optimizer=None, train_loss=None, train_accuracy=None,
                        val_loss=None, val_accuracy=None):
    settings = SETTINGS_LEAD_TRANSFORMER if network_type == NetworkType.lead else SETTINGS_ACMP_TRANSFORMER
    num_layers = settings["NUM_LAYERS"]
    d_model = settings["D_MODEL"]
    num_heads = settings["NUM_HEADS"]
    dff = settings["DFF"]

    if network_type == NetworkType.lead:
        transformer = Transformer(num_layers=num_layers,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  input_vocab_sizes=[INPUT_VOCAB_SIZE_DIF],
                                  target_vocab_size=LEAD_OUTPUT_VOCAB_SIZE,
                                  num_encoders=1,
                                  mask_types_enc=[MaskType.singleout],
                                  mask_types_dec=[MaskType.singleout],
                                  attention_type=AttentionType.relative,
                                  max_relative_distance=SEQUENCE_MAX_LENGTH)
    elif network_type == NetworkType.acmp:
        transformer = Transformer(num_layers=num_layers,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  input_vocab_sizes=[INPUT_VOCAB_SIZE_MLD, INPUT_VOCAB_SIZE_DIF],
                                  target_vocab_size=ACMP_OUTPUT_VOCAB_SIZE,
                                  num_encoders=2,
                                  mask_types_enc=[MaskType.padding, MaskType.singleout],
                                  mask_types_dec=[MaskType.padding, MaskType.singleout],
                                  attention_type=AttentionType.relative,
                                  max_relative_distance=SEQUENCE_MAX_LENGTH)
    else:
        raise NotImplementedError

    trainer = Trainer(transformer=transformer, optimizer=optimizer,
                      train_loss=train_loss, train_accuracy=train_accuracy,
                      val_loss=val_loss, val_accuracy=val_accuracy, strategy=strategy)

    return transformer, trainer


def train_network(network_type, run_identifier=None):
    logger = get_logger(__name__)

    strategy = get_strategy()

    # Create nullcontext if run in single GPU mode
    if strategy is None:
        context = nullcontext()
    else:
        logger.info("Starting in distributed mode.")
        context = strategy.scope()

    with context:
        # Load settings
        settings = SETTINGS_LEAD_TRANSFORMER if network_type == NetworkType.lead else SETTINGS_ACMP_TRANSFORMER
        d_model = settings["D_MODEL"]
        batch_size = settings["BATCH_SIZE"]

        logical_gpus = tf.config.list_logical_devices("GPU")
        logger.info(f"Running with {len(logical_gpus)} virtual GPUs...")

        logger.info("Loading dataset...")
        train_ds = load_records(input_path=[DATA_TRAIN_OUTPUT_FILE_PATH])
        val_ds = load_records(input_path=[DATA_VAL_OUTPUT_FILE_PATH])

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        train_ds = train_ds.with_options(options)
        val_ds = val_ds.with_options(options)

        # Count batches
        amount_train_batches = len(list(train_ds.batch(batch_size).as_numpy_iterator()))
        logger.info(f"Train dataset consists of {amount_train_batches} batches.")

        amount_val_batches = len(list(val_ds.batch(batch_size).as_numpy_iterator()))
        logger.info(f"Validation dataset consists of {amount_val_batches} batches.")

        amount_batches = amount_train_batches + amount_val_batches
        logger.info(f"Overall dataset consists of {amount_batches} batches.")

        logger.info("Constructing model...")

        # Load learning rate
        learning_rate = TransformerLearningRateSchedule(d_model)

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

        # Setup time and run identifier
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if run_identifier is None:
            run_identifier = current_time

        # For restarting at a later epoch
        start_epoch = tf.Variable(0)

        # Set checkpoint
        checkpoint = tf.train.Checkpoint(transformer=transformer,
                                         optimizer=optimizer)
        checkpoint_path = f"{PATH_CHECKPOINT}/{network_type.value}/{network_type.value} {run_identifier}"
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=EPOCHS)

        # If checkpoint exists, restore it
        if checkpoint_manager.latest_checkpoint:
            start_epoch = tf.Variable(len(checkpoint_manager.checkpoints))

            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            logger.info(
                f"Restored checkpoint for epoch {len(checkpoint_manager.checkpoints)}. "
                f"Will start from epoch {start_epoch.numpy() + 1}. "
                f"{optimizer.iterations.numpy()} iterations already completed.")

            train_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/train"
            val_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/val"
        else:
            train_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/train"
            val_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/val"

        # Tensorboard setup
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        logger.info("Starting training process...")
        for epoch in range(start_epoch.numpy(), EPOCHS):
            # Prepare datasets
            train_distributed_ds = train_ds \
                .cache() \
                .shuffle(BUFFER_SIZE, seed=SHUFFLE_SEED + epoch) \
                .batch(batch_size) \
                .prefetch(tf.data.AUTOTUNE)
            val_distributed_ds = val_ds \
                .cache() \
                .shuffle(BUFFER_SIZE, seed=SHUFFLE_SEED + epoch) \
                .batch(batch_size) \
                .prefetch(tf.data.AUTOTUNE)

            # Distribute datasets
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
                inputs, target = _load_data(network_type, batch)

                # Train step
                if strategy is not None:
                    trainer.distributed_train_step(inputs, target)
                else:
                    trainer.train_step(inputs, target)

                # Tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar("train_loss", train_loss.result(), step=optimizer.iterations)
                    tf.summary.scalar("train_accuracy", train_accuracy.result(), step=optimizer.iterations)

                # Logging
                mem_usage = tf.config.experimental.get_memory_info("GPU:0")
                logger.info(
                    f"[E{epoch + 1:02d}B{batch_num + 1:05d}]: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}. "
                    f"Time taken: {round(time.time() - batch_timer, 2):.2f}s ({mem_usage['peak'] / 1e+9 :.2f} GB)")

                # Reset timer
                batch_timer = time.time()
                tf.config.experimental.reset_memory_stats("GPU:0")

            logger.info(f"[E{epoch + 1:02d}]: Calculating validation statistics...")

            # Validation batches
            for (batch_num, batch) in enumerate(val_distributed_ds):
                # Load data
                inputs, target = _load_data(network_type, batch)

                # Validation step
                if strategy is not None:
                    trainer.distributed_val_step(inputs, target)
                else:
                    trainer.val_step(inputs, target)

            # Tensorboard
            with val_summary_writer.as_default():
                tf.summary.scalar("val_loss", val_loss.result(), step=epoch + 1)
                tf.summary.scalar("val_accuracy", val_accuracy.result(), step=epoch + 1)

            # Logging
            logger.info(f"[E{epoch + 1:02d}]: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}. "
                        f"Val Loss {val_loss.result():.4f}, Val Accuracy {val_accuracy.result():.4f}. "
                        f"Time taken: {round(time.time() - epoch_timer, 2)}s")

            # Save checkpoint
            checkpoint_save_path = checkpoint_manager.save()
            logger.info(f"[E{epoch + 1:02d}]: Saving checkpoint at {checkpoint_save_path}.")

        # Save model
        logger.info("Finished training process. Saving model.")
        transformer.save_weights(f"{PATH_SAVED_MODEL}/{network_type.value}/model_{run_identifier}.h5")


def generate(network_type, model_identifier, difficulty, lead_sequence=None):
    logger = get_logger(__name__)

    # Load model
    logger.info("Constructing model...")
    transformer, _ = get_network_objects(network_type)
    transformer.build_model()
    transformer.load_weights(f"{PATH_SAVED_MODEL}/{network_type.value}/model_{model_identifier}.h5")

    # Create starting sequence
    seq = Sequence()
    lead_seq = None

    if network_type == NetworkType.lead:
        seq.add_relative_message(
            Message.from_dict({"message_type": "time_signature", "numerator": 4, "denominator": 4}))
    elif network_type == NetworkType.acmp:
        lead_seq = lead_sequence

        # TODO
        lead_seq = \
        Sequence.sequences_from_midi_file(f"{get_prj_root()}/out/paul/gen/lead/3_20220530-111731.mid", [[0]], [])[0]

    generator = Generator(transformer, network_type, lead_sequence=lead_seq)

    schedule = TemperatureSchedule(96, 12, 1 / 2, exponent=2.5, max_value=1, min_value=0.2)

    sequences, attention_weights = generator(input_sequence=seq, difficulty=difficulty, temperature=0.4)

    # Construct quantisation parameters
    quantise_parameters = get_note_durations(1, 8)
    quantise_parameters += get_tuplet_durations(quantise_parameters, 3, 2)

    logger.info("Calculating difficulties...")

    deviations = []
    # Calculate difficulty deviations
    for i, sequence in enumerate(sequences):
        sequence.quantise(quantise_parameters)
        sequence.quantise_note_lengths()

        desired_difficulty = difficulty
        if desired_difficulty == DIFFICULTY_VALUE_SCALE - 1:
            desired_difficulty = DIFFICULTY_VALUE_SCALE

        output_difficulties = []

        bars = Sequence.split_into_bars([sequence])
        for bar in bars[0]:
            output_difficulties.append(convert_difficulty(bar.difficulty()))

        deviation = _deviation(output_difficulties, desired_difficulty)
        deviations.append((sequence, deviation))

    deviations = sorted(deviations, key=itemgetter(1))
    logger.info(f"Deviations: {deviations}")
    sequence = deviations[0][0]

    logger.info(f"Saving generated sequence...")

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"{PATH_MIDI}/{network_type.value}/{difficulty}_{current_time}.mid"
    sequence.save(output_path)


def store_checkpoint(network_type, run_identifier, checkpoint_identifier):
    logger = get_logger(__name__)

    # Load settings
    settings = SETTINGS_LEAD_TRANSFORMER if network_type == NetworkType.lead else SETTINGS_ACMP_TRANSFORMER
    d_model = settings["D_MODEL"]

    # Load objects
    learning_rate = TransformerLearningRateSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer, _ = get_network_objects(network_type)
    transformer.build_model()

    # Setup checkpoint
    checkpoint = tf.train.Checkpoint(transformer=transformer,
                                     optimizer=optimizer)
    checkpoint_path = f"{PATH_CHECKPOINT}/{network_type.value}/{network_type.value} {run_identifier}"
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=EPOCHS)
    checkpoint.restore(checkpoint_manager.checkpoints[checkpoint_identifier])

    transformer.save_weights(f"{PATH_SAVED_MODEL}/{network_type.value}/model_{run_identifier}.h5")

    logger.info(f"Stored model {run_identifier} from epoch {checkpoint_identifier + 1}.")


def _load_data(network_type, batch):
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

    if network_type == NetworkType.lead:
        inputs = [lead_dif]
        target = lead_seq
    elif network_type == NetworkType.acmp:
        inputs = [lead_seq, acmp_dif]
        target = acmp_seq
    else:
        raise NotImplementedError

    return inputs, target


def _deviation(values, mu, power_raise=2):
    values = tf.convert_to_tensor(values)
    mu_tensor = tf.fill([len(values)], mu)
    subtract = tf.subtract(values, mu_tensor)
    power = tf.pow(subtract, power_raise)
    red_sum = tf.reduce_sum(power)
    divide = tf.divide(red_sum, len(values))
    root = tf.pow(divide, 1 / power_raise)

    return root
