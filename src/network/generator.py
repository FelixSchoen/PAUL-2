import numpy as np
import tensorflow as tf
from sCoda import Sequence
from sCoda.elements.message import MessageType, Message

from src.config.settings import D_TYPE, START_TOKEN, SEQUENCE_MAX_LENGTH, STOP_TOKEN, SETTINGS_LEAD_TRANSFORMER, \
    SETTINGS_ACMP_TRANSFORMER, VALID_TIME_SIGNATURES
from src.preprocessing.data_pipeline import Tokenizer, Detokenizer
from src.util.enumerations import NetworkType


class Generator(tf.Module):

    def __init__(self, transformer, network_type, lead_sequence=None):
        super().__init__()
        self.transformer = transformer
        self.network_type = network_type
        self.lead_sequence = lead_sequence

    @tf.function
    def new_call(self, start_sequence, difficulty, temperature=0, bars=4):
        assert not (self.network_type == NetworkType.acmp and self.lead_sequence is None)
        assert len(start_sequence._get_rel().messages) <= SEQUENCE_MAX_LENGTH - 2

        # Initialise objects
        output_sequence = Sequence()
        tokenizer = Tokenizer()
        detokenizer = Detokenizer()
        settings = SETTINGS_LEAD_TRANSFORMER if self.network_type == NetworkType.lead else SETTINGS_ACMP_TRANSFORMER

        # Add start sequence to output sequence
        output_sequence.merge([start_sequence])

        # Build difficulty tensor
        dif_value = tokenizer.tokenize_difficulty((difficulty - 1) / 10)
        dif_tensor_array = tf.TensorArray(D_TYPE, size=SEQUENCE_MAX_LENGTH, dynamic_size=False)
        for i in range(1, SEQUENCE_MAX_LENGTH - 1):
            dif_tensor_array = dif_tensor_array.write(i, dif_value)
        dif_tensor_array = dif_tensor_array.write(0, START_TOKEN)
        dif_tensor_array = dif_tensor_array.write(SEQUENCE_MAX_LENGTH - 1, STOP_TOKEN)
        dif_tensor = tf.expand_dims(dif_tensor_array.stack(), 0)

        # Build lead tensor
        if self.network_type == NetworkType.acmp:
            lead_tensor_array = Generator._create_tensor_from_sequence(self.lead_sequence, write_stop_token=True)

        # Build output tensor
        output_tensor_array = Generator._create_tensor_from_sequence(start_sequence, write_stop_token=False)

        # Create input tensors
        if self.network_type == NetworkType.lead:
            input_tensors = [dif_tensor]
        elif self.network_type == NetworkType.acmp:
            raise NotImplementedError
        else:
            raise NotImplementedError

        # Loop for up to remaining tokens time
        for i in tf.range(1 + len(tokens), SEQUENCE_MAX_LENGTH - 1):
            output_tensor = tf.expand_dims(output_tensor_array.stack(), 0)

            # Interference
            predictions, _ = self.transformer([input_tensors, output_tensor], training=False)
            prediction_tensor = predictions[:, i]  # TODO Different, uses -1 instead of i, makes no sense to me

            # Create and apply valid messages mask
            valid_next_messages = output_sequence._get_rel().get_valid_next_messages(desired_bars=bars,
                                                                                     force_time_siganture=self.network_type == NetworkType.lead)
            mask = Generator._create_mask_from_valid_messages(valid_next_messages,
                                                              mask_length=settings["OUTPUT_SIZE"]) * -np.inf
            mask = tf.where(tf.math.is_nan(mask), tf.zeros_like(mask), mask)
            prediction_tensor += mask

            # Determine prediction
            if temperature == 0:
                prediction = tf.argmax(prediction_tensor, axis=-1)[0]
            else:
                prediction = tf.random.categorical(prediction_tensor / temperature, 1)[0][0]

            # Write prediction to output tensor
            output_tensor_array.write(i + 1, prediction)

            # Add to sequence
            detokens = detokenizer.detokenize(prediction)
            detokens.extend(detokenizer.flush_wait_buffer())
            for detoken in detokens:
                output_sequence.add_relative_message(Message.from_dict(detoken))

            break

        # Calculate attention
        output_tensor = tf.expand_dims(output_tensor_array.stack(), 0)
        _, attention_weights = self.transformer([input_tensors, output_tensor], training=False)

        return output_tensor, attention_weights

    @tf.function
    def __call__(self, input_tensors, temperature=1):
        for input_tensor in input_tensors:
            assert isinstance(input_tensor, tf.Tensor)

        output_tensor_array = tf.TensorArray(D_TYPE, size=SEQUENCE_MAX_LENGTH - 1, dynamic_size=False)
        output_tensor_array = output_tensor_array.write(0, START_TOKEN)

        # Loop
        for i in tf.range(SEQUENCE_MAX_LENGTH - 1):
            output_tensor = tf.expand_dims(output_tensor_array.stack(), 0)

            predictions, _ = self.transformer([input_tensors, output_tensor], training=False)
            prediction_tensor = predictions[:, i]  # TODO Different, uses -1 instead of i, makes no sense to me

            if temperature == 0:
                prediction = tf.argmax(prediction_tensor, axis=-1)[0]
            else:
                prediction = tf.random.categorical(prediction_tensor / temperature, 1)[0][0]

            output_tensor_array.write(i + 1, prediction)

        # Calculate attention
        output_tensor = tf.expand_dims(output_tensor_array.stack(), 0)
        _, attention_weights = self.transformer([input_tensors, output_tensor], training=False)

        return output_tensor, attention_weights

    @staticmethod
    def _create_tensor_from_sequence(sequence, write_stop_token=False):
        tokenizer = Tokenizer()
        end_index = 0

        data_frame = sequence.to_relative_dataframe()
        tensor_array = tf.TensorArray(D_TYPE, size=SEQUENCE_MAX_LENGTH, dynamic_size=False)
        tensor_array = tensor_array.write(0, START_TOKEN)
        tokens = []
        for _, row in data_frame.iterrows():
            tokens.extend(tokenizer.tokenize(row))
        tokens.extend(tokenizer.flush_wait_buffer())
        for i, token in enumerate(tokens):
            tensor_array = tensor_array.write(i + 1, token)
            end_index = i + 2

        if write_stop_token:
            tensor_array = tensor_array.write(end_index, STOP_TOKEN)

        return tensor_array

    @staticmethod
    def _create_mask_from_valid_messages(valid_messages, mask_length):
        tokenizer = Tokenizer()

        mask = [1 for _ in range(mask_length)]
        messages = []
        tokens = []

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

        for message in messages:
            tokens.extend(tokenizer.tokenize(message))
            tokens.extend(tokenizer.flush_wait_buffer())

        for token in tokens:
            if token < len(mask):
                mask[token] = 0

        return tf.convert_to_tensor(mask, dtype=tf.dtypes.float32)
