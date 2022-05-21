import multiprocessing

import numpy as np
import tensorflow as tf
from sCoda import Sequence
from sCoda.elements.message import MessageType, Message

from src.config.settings import D_TYPE, START_TOKEN, SEQUENCE_MAX_LENGTH, STOP_TOKEN, SETTINGS_LEAD_TRANSFORMER, \
    SETTINGS_ACMP_TRANSFORMER, VALID_TIME_SIGNATURES, OUTPUT_DIMENSION, PADDING_TOKEN, CONSECUTIVE_BAR_MAX_LENGTH
from src.preprocessing.preprocessing import Tokenizer, Detokenizer
from src.util.enumerations import NetworkType
from src.util.logging import get_logger


class Generator(tf.Module):

    def __init__(self, transformer, network_type, lead_sequence=None):
        super().__init__()
        self.transformer = transformer
        self.network_type = network_type
        self.lead_sequence = lead_sequence

    def __call__(self, input_sequence, difficulty, temperature, bars=CONSECUTIVE_BAR_MAX_LENGTH):
        logger = get_logger(__name__)

        assert not (self.network_type == NetworkType.acmp and self.lead_sequence is None)

        # Initialise objects
        output_sequences = [Sequence() for _ in range(OUTPUT_DIMENSION)]
        finished_sequence_indices = []
        tokenizer = Tokenizer()
        detokenizer = Detokenizer()
        settings = SETTINGS_LEAD_TRANSFORMER if self.network_type == NetworkType.lead else SETTINGS_ACMP_TRANSFORMER

        # Add start sequence to output sequence
        for output_sequence in output_sequences:
            output_sequence.merge([input_sequence])

        # Build difficulty tensors
        dif_tensor_arrays = []
        dif_value = tokenizer.tokenize_difficulty(difficulty / 10)

        for d in range(OUTPUT_DIMENSION):
            dif_tensor_array = tf.TensorArray(D_TYPE, size=SEQUENCE_MAX_LENGTH, dynamic_size=False)
            for i in range(1, SEQUENCE_MAX_LENGTH - 1):
                dif_tensor_array = dif_tensor_array.write(i, dif_value)
            dif_tensor_array = dif_tensor_array.write(0, START_TOKEN)
            dif_tensor_array = dif_tensor_array.write(SEQUENCE_MAX_LENGTH - 1, STOP_TOKEN)

            dif_tensor_arrays.append(dif_tensor_array)
        dif_tensor = Generator._create_tensor_from_tensor_arrays(dif_tensor_arrays)

        # Build lead tensor
        if self.network_type == NetworkType.acmp:
            lead_tensor_arrays, _ = Generator._create_tensor_arrays_from_sequence(self.lead_sequence,
                                                                                  write_stop_token=True)
            lead_tensor = Generator._create_tensor_from_tensor_arrays(lead_tensor_arrays)

        # Build output tensor
        output_tensor_arrays, len_input_seq = Generator._create_tensor_arrays_from_sequence(input_sequence,
                                                                                            size=SEQUENCE_MAX_LENGTH - 1,
                                                                                            write_stop_token=False)

        # Create input tensors
        if self.network_type == NetworkType.lead:
            input_tensors = [dif_tensor]
        elif self.network_type == NetworkType.acmp:
            input_tensors = [lead_tensor, dif_tensor]
        else:
            raise NotImplementedError

        logger.info("Starting generation process")

        # TODO Remove
        time_tracker = 0

        pool = multiprocessing.Pool()
        bars_list = [bars for _ in range(OUTPUT_DIMENSION)]
        force_time_signature_list = [self.network_type == NetworkType.lead for _ in range(OUTPUT_DIMENSION)]

        # Loop for up to remaining tokens time
        for i in tf.range(1 + len_input_seq, SEQUENCE_MAX_LENGTH - 1):
            if i % 5 == 0 or i == 1 + len_input_seq:
                logger.info(f"Iteration {i:03d}...")

            output_tensor = Generator._create_tensor_from_tensor_arrays(output_tensor_arrays)

            # Interference
            predictions, _ = self.transformer([input_tensors, output_tensor], training=False)
            prediction_tensor = predictions[:, i - 1]

            # Create and apply valid messages mask
            valid_messages_list = list(pool.starmap(Generator._create_valid_messages_from_sequence,
                                                    zip(output_sequences, bars_list, force_time_signature_list)))

            mask = Generator._create_mask_from_valid_messages(valid_messages_list,
                                                              mask_length=settings["OUTPUT_SIZE"],
                                                              finished_sequence_indices=finished_sequence_indices) * -np.inf
            mask = tf.where(tf.math.is_nan(mask), tf.zeros_like(mask), mask)
            prediction_tensor += mask

            if not callable(temperature):
                temperature_function = lambda _: temperature
            else:
                temperature_function = temperature

            # Determine prediction
            if temperature_function(time_tracker) == 0:
                predictions = tf.argmax(prediction_tensor, axis=-1)
            else:
                predictions = tf.squeeze(
                    tf.random.categorical(prediction_tensor / temperature_function(time_tracker), 1), [-1])

            predictions = tf.cast(predictions, dtype=D_TYPE)

            for d in range(OUTPUT_DIMENSION):
                output_sequence = output_sequences[d]
                output_tensor_array = output_tensor_arrays[d]
                prediction = predictions[d]

                # Write prediction to output tensor
                output_tensor_array.write(i + 1, prediction)

                # Check if end of sequence was predicted
                if prediction.numpy() == STOP_TOKEN:
                    finished_sequence_indices.append(d)

                # Add predicted message to sequence
                detokens = detokenizer.detokenize(prediction.numpy())
                detokens.extend(detokenizer.flush_wait_buffer())
                for detoken in detokens:
                    msg = Message.from_dict(detoken)
                    output_sequence.add_relative_message(msg)

                    if msg.message_type == MessageType.wait:
                        time_tracker += msg.time

            # Check if all sequences finished
            if len(finished_sequence_indices) == OUTPUT_DIMENSION:
                break

        # Assert that all sequences have finished properly
        assert len(finished_sequence_indices) == OUTPUT_DIMENSION

        # Calculate attention
        output_tensor = Generator._create_tensor_from_tensor_arrays(output_tensor_arrays)
        _, attention_weights = self.transformer([input_tensors, output_tensor], training=False)

        return output_sequences, attention_weights

    @staticmethod
    def _create_tensor_arrays_from_sequence(sequence, size=SEQUENCE_MAX_LENGTH, write_stop_token=False):
        tokenizer = Tokenizer()
        end_index = 0

        tensor_arrays = []

        for i in range(OUTPUT_DIMENSION):
            tensor_array = tf.TensorArray(D_TYPE, size=size, dynamic_size=False)
            tensor_array = tensor_array.write(0, START_TOKEN)
            tensor_arrays.append(tensor_array)

        data_frame = sequence.to_relative_dataframe()
        tokens = []
        for _, row in data_frame.iterrows():
            tokens.extend(tokenizer.tokenize(row))
        tokens.extend(tokenizer.flush_wait_buffer())

        assert len(tokens) <= SEQUENCE_MAX_LENGTH - 2

        for i, token in enumerate(tokens):
            for j, tensor_array in enumerate(tensor_arrays):
                tensor_arrays[j] = tensor_array.write(i + 1, token)
                end_index = i + 2

        if write_stop_token:
            for j, tensor_array in tensor_arrays:
                tensor_arrays[j] = tensor_array.write(end_index, STOP_TOKEN)

        return tensor_arrays, len(tokens)

    @staticmethod
    def _create_tensor_from_tensor_arrays(tensor_arrays):
        tensor = tf.stack([tensor_array.stack() for tensor_array in tensor_arrays])
        return tensor

    @staticmethod
    def _create_valid_messages_from_sequence(sequence, desired_bars, force_time_signature):
        return sequence._get_rel().get_valid_next_messages(desired_bars=desired_bars,
                                                           force_time_siganture=force_time_signature)

    @staticmethod
    def _create_mask_from_valid_messages(valid_messages_list, mask_length, finished_sequence_indices):
        # Initialise masks
        mask = [[1 for _ in range(mask_length)] for _ in range(OUTPUT_DIMENSION)]

        for i in range(OUTPUT_DIMENSION):
            tokenizer = Tokenizer()
            valid_messages = valid_messages_list[i]
            messages = []
            tokens = []

            # Handle messages that represent multiple valid outcomes (e.g., wait)
            for msg in valid_messages:
                if msg["message_type"] == MessageType.wait.value:
                    # Append all types of messages
                    for wait_time in range(1, 25):
                        messages.append({"message_type": "wait", "time": wait_time})
                elif msg["message_type"] == MessageType.time_signature.value:
                    for signature in VALID_TIME_SIGNATURES:
                        messages.append(
                            {"message_type": "time_signature", "numerator": signature[0], "denominator": signature[1]})
                else:
                    tokens.extend(tokenizer.tokenize(msg))

            # Tokenize messages
            for message in messages:
                tokens.extend(tokenizer.tokenize(message))
                tokens.extend(tokenizer.flush_wait_buffer())

            # Allow tokens
            for token in tokens:
                if token < mask_length:
                    mask[i][token] = 0

            # Check if no more valid messages
            if len(tokens) == 0:
                if i in finished_sequence_indices:
                    mask[i][PADDING_TOKEN] = 0
                else:
                    mask[i][STOP_TOKEN] = 0

        return tf.convert_to_tensor(mask, dtype=tf.dtypes.float32)


class TemperatureSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, max_time, warmup_steps, warmup_multiplier, exponent, max_value, min_value) -> None:
        super(TemperatureSchedule, self).__init__()

        self.a = max_time
        self.warmup_steps = warmup_steps
        self.warmup_multiplier = warmup_multiplier
        self.n = exponent
        self.b = max_value
        self.c = min_value

    def __call__(self, step):
        i_r = tf.math.divide(tf.math.subtract(1, tf.math.pow(tf.math.divide(step, self.a), self.n)), self.b)
        i_r = tf.math.maximum(0, i_r)

        warmup_value = ((1 / self.b) * self.warmup_multiplier)
        step_size = warmup_value / self.warmup_steps
        i_r_w = warmup_value + step_size * step

        return tf.cast(tf.math.maximum(tf.math.minimum(i_r, i_r_w), self.c), dtype=tf.dtypes.float32)
