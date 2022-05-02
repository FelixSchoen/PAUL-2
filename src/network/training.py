import tensorflow as tf

from src.network.optimization import accuracy_function, loss_function
from src.config.settings import D_TYPE


class Trainer:
    strategy, transformer, optimizer, train_loss, train_accuracy, val_loss, val_accuracy, mask_types, signature = \
        None, None, None, None, None, None, None, None, None

    def __init__(self, transformer, optimizer, train_loss, train_accuracy, val_loss, val_accuracy, mask_types, *,
                 strategy):
        Trainer.strategy = strategy
        Trainer.transformer = transformer
        Trainer.optimizer = optimizer
        Trainer.train_loss = train_loss
        Trainer.train_accuracy = train_accuracy
        Trainer.val_loss = val_loss
        Trainer.val_accuracy = val_accuracy
        Trainer.mask_types = mask_types

        Trainer.signature = [
            tf.TensorSpec(shape=(None,), dtype=D_TYPE),
            tf.TensorSpec(shape=(None,), dtype=D_TYPE),
        ]

    @staticmethod
    @tf.function  # (input_signature=signature)
    def train_step(inputs, target):
        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = Trainer.transformer([inputs, tar_inp],
                                                 training=True,
                                                 mask_types=Trainer.mask_types)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, Trainer.transformer.trainable_variables)
        Trainer.optimizer.apply_gradients(zip(gradients, Trainer.transformer.trainable_variables))

        Trainer.train_loss(loss)
        Trainer.train_accuracy(accuracy_function(tar_real, predictions))

        return loss

    @staticmethod
    @tf.function
    def val_step(inputs, target):
        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        predictions, _ = Trainer.transformer([inputs, tar_inp],
                                             training=True,
                                             mask_types=Trainer.mask_types)
        loss = loss_function(tar_real, predictions)

        Trainer.val_loss(loss)
        Trainer.val_accuracy(accuracy_function(tar_real, predictions))

        return loss

    @staticmethod
    @tf.function
    def distributed_train_step(inputs, target):
        per_replica_losses = Trainer.strategy.run(Trainer.train_step,
                                                  args=(inputs, target))
        overall_loss = Trainer.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        Trainer.train_loss(overall_loss)

        return overall_loss

    @staticmethod
    @tf.function
    def distributed_val_step(inputs, target):
        per_replica_losses = Trainer.strategy.run(Trainer.val_step,
                                                  args=(inputs, target))
        overall_loss = Trainer.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        Trainer.val_loss(overall_loss)

        return overall_loss
