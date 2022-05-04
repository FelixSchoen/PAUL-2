import tensorflow as tf

from src.config.settings import D_TYPE, START_TOKEN, SEQUENCE_MAX_LENGTH


class Generator(tf.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    @tf.function
    def __call__(self, input_tensors, temperature=0):
        for input_tensor in input_tensors:
            assert isinstance(input_tensor, tf.Tensor)

        output_tensor_array = tf.TensorArray(D_TYPE, size=SEQUENCE_MAX_LENGTH - 1, dynamic_size=False)
        output_tensor_array = output_tensor_array.write(0, START_TOKEN)

        # Loop
        for i in tf.range(SEQUENCE_MAX_LENGTH - 1):
            output_tensor = tf.expand_dims(output_tensor_array.stack(), 0)

            predictions, _ = self.transformer([input_tensors, output_tensor], training=False)
            prediction_tensor = predictions[:, i]  # TODO Different, uses -1

            if temperature == 0:
                prediction = tf.argmax(prediction_tensor, axis=-1)[0]
            else:
                prediction = tf.random.categorical(prediction_tensor / temperature, 1)[0][0]

            output_tensor_array.write(i + 1, prediction)

        # Calculate attention
        output_tensor = tf.expand_dims(output_tensor_array.stack(), 0)
        _, attention_weights = self.transformer([input_tensors, output_tensor], training=False)

        return output_tensor, attention_weights
