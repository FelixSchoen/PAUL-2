import tensorflow as tf

from src.network.optimization import loss_function, accuracy_function


class Trainer:
    signature = None

    def __init__(self, transformer, optimizer, train_loss, train_accuracy) -> None:
        super().__init__()
        self.transformer = transformer
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy

    def __call__(self, *args, **kwargs):
        self.train_step(self.transformer, self.optimizer, self.train_loss, self.train_accuracy, args[0], args[1])

    @staticmethod
    @tf.function
    def train_step(transformer, optimizer, train_loss, train_accuracy, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))
