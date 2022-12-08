import os
import time
from contextlib import nullcontext
from datetime import datetime
from math import floor
from pathlib import Path

import tensorflow as tf
from sCoda import Sequence, Message, Bar
from sCoda.elements.message import MessageType
from tensorflow.python.data.ops.options import AutoShardPolicy

from src.config.settings import LEAD_OUTPUT_VOCAB_SIZE, \
    INPUT_VOCAB_SIZE_DIF, PATH_CHECKPOINT, BUFFER_SIZE, SHUFFLE_SEED, SEQUENCE_MAX_LENGTH, EPOCHS, \
    PATH_TENSORBOARD, ACMP_OUTPUT_VOCAB_SIZE, INPUT_VOCAB_SIZE_MLD, PATH_SAVED_MODEL, DATA_TRAIN_OUTPUT_FILE_PATH, \
    DATA_VAL_OUTPUT_FILE_PATH, SETTINGS_LEAD_TRANSFORMER, SETTINGS_ACMP_TRANSFORMER, VAL_PER_BATCHES, \
    MAX_CHECKPOINTS_TO_KEEP, \
    MAXIMUM_NOTE_LENGTH, BARS_TO_GENERATE, BAR_GENERATION_STEP_SIZE, PATH_MIDI
from src.network.attention import AttentionType
from src.network.generator import Generator
from src.network.masking import MaskType
from src.network.optimization import TransformerLearningRateSchedule
from src.network.training import Trainer
from src.network.transformer import Transformer
from src.preprocessing.preprocessing import load_records
from src.util.enumerations import NetworkType
from src.util.logging import get_logger
from src.util.util import get_src_root, get_prj_root, convert_difficulty


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
                                  mask_types=[[MaskType.singleout],
                                              [MaskType.lookahead, MaskType.singleout]],
                                  attention_types=[[AttentionType.absolute],
                                                   [AttentionType.self_relative, AttentionType.absolute]],
                                  max_relative_distance=SEQUENCE_MAX_LENGTH)
    elif network_type == NetworkType.acmp:
        transformer = Transformer(num_layers=num_layers,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  input_vocab_sizes=[INPUT_VOCAB_SIZE_DIF, INPUT_VOCAB_SIZE_MLD],
                                  target_vocab_size=ACMP_OUTPUT_VOCAB_SIZE,
                                  num_encoders=2,
                                  mask_types=[[MaskType.singleout],
                                              [MaskType.padding],
                                              [MaskType.lookahead, MaskType.singleout, MaskType.padding, ]],
                                  attention_types=[[AttentionType.absolute],
                                                   [AttentionType.absolute],
                                                   [AttentionType.self_relative, AttentionType.absolute,
                                                    AttentionType.absolute]],
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
        val_per_epoch = batch_size * VAL_PER_BATCHES

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

        # amount_train_samples = batch_size * amount_train_batches
        # logger.info(f"Train dataset consists of {amount_train_samples} samples.")
        #
        # amount_val_samples = batch_size * amount_val_batches
        # logger.info(f"Validation dataset consists of {amount_val_samples} samples.")

        logger.info(f"Creating a checkpoint every {int(amount_train_batches / val_per_epoch)} batches.")

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
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path,
                                                        max_to_keep=MAX_CHECKPOINTS_TO_KEEP)

        # If checkpoint exists, restore it
        completed_batches_of_epoch = 0
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)

            completed_epochs = floor(optimizer.iterations.numpy() / amount_train_batches)
            completed_batches_of_epoch = optimizer.iterations.numpy() - (amount_train_batches * completed_epochs)

            start_epoch = tf.Variable(completed_epochs)

            logger.info(f"Restored checkpoint with {completed_epochs} epochs completed. "
                        f"Completed {completed_batches_of_epoch} batches of current epoch. "
                        f"Completed {optimizer.iterations.numpy()} steps so far.")

            train_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/train"
            val_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/val"
        else:
            train_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/train"
            val_log_dir = f"{PATH_TENSORBOARD}/{network_type.value}/{run_identifier}/val"

        # Save settings to file
        try:
            Path(checkpoint_path).mkdir(exist_ok=True)
            with open(f"{checkpoint_path}/settings.txt", "x") as f:
                print(f"Batches: {amount_batches}", file=f)
                print(f"Train Batches: {amount_train_batches}", file=f)
                print(f"Validation Batches: {amount_val_batches}", file=f)
                print(f"Layers: {settings['NUM_LAYERS']}", file=f)
                print(f"Heads: {settings['NUM_HEADS']}", file=f)
                print(f"Dimension Model: {settings['D_MODEL']}", file=f)
                print(f"Dimension Feed Forward Network: {settings['DFF']}", file=f)
                print(f"Dropout: {settings['DROPOUT_RATE']}", file=f)
                print(f"Batch Size: {settings['BATCH_SIZE']}", file=f)
        except FileExistsError:
            pass

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

            # Batches
            for (batch_num, batch) in enumerate(train_distributed_ds):
                if completed_batches_of_epoch > 0:
                    completed_batches_of_epoch -= 1
                    continue

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

                # Validation step
                if optimizer.iterations.numpy() % floor(amount_train_batches / val_per_epoch) == 0 \
                        and optimizer.iterations.numpy() != 0:
                    logger.info(
                        f"[E{epoch + 1:02d}B{batch_num + 1:05d}S{optimizer.iterations.numpy():05d}]: "
                        f"Calculating validation statistics...")

                    # Validation batches
                    for (val_batch_num, val_batch) in enumerate(val_distributed_ds):
                        # Load data
                        inputs, target = _load_data(network_type, val_batch)

                        # Validation step
                        if strategy is not None:
                            trainer.distributed_val_step(inputs, target)
                        else:
                            trainer.val_step(inputs, target)

                    step = floor(optimizer.iterations.numpy() / floor(amount_train_batches / val_per_epoch))

                    # Tensorboard
                    with val_summary_writer.as_default():
                        tf.summary.scalar("val_loss", val_loss.result(), step=step)
                        tf.summary.scalar("val_accuracy", val_accuracy.result(), step=step)

                    # Validation Logging
                    logger.info(
                        f"[E{epoch + 1:02d}]: Val Loss {val_loss.result():.4f}, Val Accuracy {val_accuracy.result():.4f}. "
                        f"Time taken: {round(time.time() - epoch_timer, 2)}s")

                    val_loss.reset_states()
                    val_accuracy.reset_states()

                    # Save checkpoint
                    checkpoint_save_path = checkpoint_manager.save(checkpoint_number=step)
                    logger.info(f"[E{epoch + 1:02d}]: Saving checkpoint at {checkpoint_save_path}.")

                # Logging
                mem_usage = tf.config.experimental.get_memory_info("GPU:0")
                if (batch_num + 1) % 50 == 0:
                    logger.info(
                        f"[E{epoch + 1:02d}B{batch_num + 1:05d}S{optimizer.iterations.numpy():05d}]: "
                        f"Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}. "
                        f"Time taken: {round(time.time() - batch_timer, 2):.2f}s ({mem_usage['peak'] / 1e+9 :.2f} GB)")

                # Reset timer
                batch_timer = time.time()
                tf.config.experimental.reset_memory_stats("GPU:0")

        # Save model
        logger.info("Finished training process. Saving model.")
        transformer.save_weights(f"{PATH_SAVED_MODEL}/{network_type.value}/model_{run_identifier}.h5")


def generate(network_type: NetworkType, model_identifier: str, difficulty: int,
             primer_sequence: Sequence = None, lead_seq: Sequence = None, name: str = ""):
    logger = get_logger(__name__)

    assert not network_type == NetworkType.acmp or lead_seq is not None

    # Load model
    logger.info("Constructing model...")
    transformer, _ = get_network_objects(network_type)
    transformer.build_model()
    transformer.load_weights(f"{PATH_SAVED_MODEL}/{network_type.value}/model_{model_identifier}.h5")
    settings = SETTINGS_LEAD_TRANSFORMER if network_type == NetworkType.lead else SETTINGS_ACMP_TRANSFORMER

    # Create sequence object
    gen_seq = primer_sequence if primer_sequence is not None else Sequence()

    # Start with 4/4 time signature if not provided
    if primer_sequence is None:
        gen_seq.rel.messages.append(Message(message_type=MessageType.time_signature, numerator=4, denominator=4))

    # Load generator
    generator = Generator(transformer, network_type, lead_sequence=lead_seq)

    # Keep track of how many valid bars have been generated so far
    valid_bars_generated = 0
    amount_bars_primer_sequence = len(Sequence.split_into_bars([gen_seq])[0])
    valid_bars_generated += amount_bars_primer_sequence

    # Check if last bar is filled to capacity, if not reduce number of valid bars generated
    if (gen_seq.sequence_length() - sum(
            bar.sequence.sequence_length() for bar in Sequence.split_into_bars([gen_seq])[0])) < 0:
        valid_bars_generated -= 1

    max_discrepancy = 1

    # Generate bars until desired number of bars has been generated
    while valid_bars_generated < BARS_TO_GENERATE:
        step_size = min(BAR_GENERATION_STEP_SIZE, BARS_TO_GENERATE - valid_bars_generated)
        temperature = settings["TEMP"]
        iteration = 0

        # Generate until all the new bars conform to the difficulty
        while True:
            logger.info(
                f"Generating bars {valid_bars_generated + 1} through {valid_bars_generated + step_size} of difficulty {difficulty}, iteration {iteration + 1}.")
            sequences, attention_weights = generator(input_sequence=gen_seq, difficulty=difficulty,
                                                     temperature=temperature,
                                                     bars_to_generate=valid_bars_generated + step_size)

            # Store index, consecutive bars of best found sequence so far
            best_sequence = (-1, 0)

            # Check all generated sequences
            for i, sequence in enumerate(sequences):
                # Quantise sequence
                sequence.quantise()
                sequence.quantise_note_lengths()
                sequence.cutoff(2 * MAXIMUM_NOTE_LENGTH, MAXIMUM_NOTE_LENGTH)

                # Get difficulties of generated bars
                output_difficulties = []

                # Obtain newly generated bars
                bars = Sequence.split_into_bars([sequence])
                new_bars = bars[0][valid_bars_generated:]
                new_bars = new_bars[:step_size]

                # Calculate difficulties of new bars
                for bar in new_bars:
                    if bar.is_empty():
                        output_difficulties.append(-1)
                    else:
                        output_difficulties.append(convert_difficulty(bar.difficulty()))

                # Calculate how many consecutive bar fall into valid range of difficulties
                consecutive_good_bars = 0
                for output_difficulty in output_difficulties:
                    if difficulty - max_discrepancy <= output_difficulty <= difficulty + max_discrepancy:
                        consecutive_good_bars += 1
                    else:
                        break

                if consecutive_good_bars > best_sequence[1]:
                    best_sequence = (i, consecutive_good_bars)
                    logger.info(
                        f"Found better sequence with {consecutive_good_bars} bar(s) and difficulties {output_difficulties}.")

            if best_sequence[0] > -1:
                valid_bars_generated += best_sequence[1]

                gen_part = Sequence.split_into_bars([sequences[best_sequence[0]]])[0][:valid_bars_generated]
                gen_seq = Bar.to_sequence(gen_part)

                break

            # Adjust step size, temperature, iteration
            if step_size % 2 == 0:
                step_size = int(step_size / 2)
            temperature *= 1.25
            iteration += 1

    logger.info(f"Saving generated sequence...")

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = f"{PATH_MIDI}/{network_type.value}/{current_time}{'_N' if name != '' else ''}{name}_{difficulty}.mid"
    gen_seq.save(output_path)


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
    index_of = next(i for i, v in enumerate(checkpoint_manager.checkpoints) if f"ckpt-{checkpoint_identifier}" in v)
    checkpoint.restore(checkpoint_manager.checkpoints[index_of])

    transformer.save_weights(f"{PATH_SAVED_MODEL}/{network_type.value}/model_{run_identifier}.h5")

    logger.info(f"Stored model {run_identifier} from epoch {checkpoint_identifier}.")


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
        inputs = [acmp_dif, lead_seq]
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
