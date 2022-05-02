import tensorflow as tf

from src.config.settings import D_TYPE, START_TOKEN, SEQUENCE_MAX_LENGTH


class Generator(tf.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def __call__(self, input_tensors, temperature=0):
        for input_tensor in input_tensors:
            assert isinstance(input_tensor, tf.Tensor)

        output = [START_TOKEN]

        # Loop
        for i in tf.range(SEQUENCE_MAX_LENGTH - 510):  # TODO
            output_padded = output + [0] * (SEQUENCE_MAX_LENGTH - 1 - len(output))
            output_tensor = tf.convert_to_tensor(output_padded, dtype=D_TYPE)
            output_tensor = tf.expand_dims(output_tensor, 0)

            predictions, _ = self.transformer([input_tensors, output_tensor], training=False)
            prediction_tensor = predictions[:, i]  # TODO Different, uses -1

            if temperature == 0:
                prediction = tf.argmax(prediction_tensor, axis=-1)
            else:
                prediction = tf.random.categorical(prediction_tensor / temperature, 1)[0]

            output.append(prediction[0])

        # Calculate attention
        output_padded = output + [0] * (SEQUENCE_MAX_LENGTH - 1 - len(output))
        output_tensor = tf.convert_to_tensor(output_padded, dtype=D_TYPE)
        output_tensor = tf.expand_dims(output_tensor, 0)
        _, attention_weights = self.transformer([input_tensors, output_tensor], training=False)
