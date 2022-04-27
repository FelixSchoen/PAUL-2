import tensorflow as tf

from src.network.optimization import accuracy_function, loss_function


class Trainer:
    transformer, optimizer, train_loss, train_accuracy, mask_types, signature = None, None, None, None, None, None

    def __init__(self, transformer, optimizer, train_loss, train_accuracy, mask_types):
        Trainer.transformer = transformer
        Trainer.optimizer = optimizer
        Trainer.train_loss = train_loss
        Trainer.train_accuracy = train_accuracy
        Trainer.mask_types = mask_types

        Trainer.signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]

    @staticmethod
    @tf.function(input_signature=signature)
    def train_step(inputs, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = Trainer.transformer([inputs, tar_inp],
                                                 training=True,
                                                 mask_types=Trainer.mask_types)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, Trainer.transformer.trainable_variables)
        Trainer.optimizer.apply_gradients(zip(gradients, Trainer.transformer.trainable_variables))

        Trainer.train_loss(loss)
        Trainer.train_accuracy(accuracy_function(tar_real, predictions))
