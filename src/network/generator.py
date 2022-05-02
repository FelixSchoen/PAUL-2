import tensorflow as tf

from src.config.settings import D_TYPE, START_TOKEN, SEQUENCE_MAX_LENGTH


class Generator(tf.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def __call__(self, input_tensors):
        for input_tensor in input_tensors:
            assert isinstance(input_tensor, tf.Tensor)

        output = [0 for _ in range(SEQUENCE_MAX_LENGTH - 1)]
        output[0] = START_TOKEN

        for i in range(1):
            output_tensor = tf.convert_to_tensor(output, dtype=D_TYPE)
            output_tensor = tf.expand_dims(output_tensor, 0)

            predictions, _ = self.transformer([input_tensors, output_tensor], training=False)
            prediction_tensor = predictions[0][i]  # TODO Different, use -1
            prediction = tf.argmax(prediction_tensor)  # TODO Categorical sampling

            print(prediction)
