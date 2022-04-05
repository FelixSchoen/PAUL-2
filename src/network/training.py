import tensorflow as tf

from src.network.optimization import loss_function, accuracy_function


class Trainer:
    signature = None

    def __init__(self, transformer, optimizer, train_loss, train_accuracy, signature) -> None:
        super().__init__()
        self.transformer = transformer
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        Trainer.signature = signature

    def __call__(self, *args, **kwargs):
        self.train_step(args[0], args[1])

    @tf.function #(input_signature=signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer([inp, tar_inp], training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions))
