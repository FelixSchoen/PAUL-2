import tensorflow as tf

from src.exception.exceptions import UnexpectedValueException
from src.network.masking import create_padding_mask, create_combined_mask, MaskType
from src.network.optimization import loss_function, accuracy_function
from src.util.logging import get_logger


class Trainer:

    def __init__(self, strategy, transformer, optimizer, train_loss, train_accuracy) -> None:
        super().__init__()
        self.strategy = strategy
        self.transformer = transformer
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy

    def __call__(self, *args, **kwargs):
        self.train_step(self.transformer, self.optimizer, self.train_loss,
                                    self.train_accuracy, mask_types=args[2], inputs=args[0], target=args[1])
        # self.distributed_train_step(self.strategy, self.transformer, self.optimizer, self.train_loss,
        #                             self.train_accuracy, mask_types=args[2], inputs=args[0], target=args[1])

    @staticmethod
    # @tf.function
    def train_step(transformer, optimizer, train_loss, train_accuracy, mask_types, inputs, target):
        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        enc_masks = []
        dec_masks = [create_combined_mask(tar_inp)]

        # Create masks
        for i, s_inp in enumerate(inputs):
            enc_masks.append(create_padding_mask(s_inp))
            if mask_types[i] == MaskType.padding:
                dec_masks.append(create_padding_mask(s_inp))
            elif mask_types[i] == MaskType.lookahead:
                dec_mask = create_combined_mask(s_inp)
                dec_mask = dec_mask[:, :, 1:, :]
                dec_masks.append(dec_mask)
            else:
                raise UnexpectedValueException("Unknown mask type")

        with tf.GradientTape() as tape:
            logger = get_logger(__name__)

            predictions, _ = transformer([inputs, tar_inp], enc_masks, dec_masks, training=True)
            loss = loss_function(tar_real, predictions)

            logger.info(f"Shape of target: {tf.shape(tar_real)}")
            logger.info(f"Shape of output: {tf.shape(predictions)}")
            logger.info(f"Loss output: {loss}")

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

        return loss

    @staticmethod
    # @tf.function
    def distributed_train_step(strategy, transformer, optimizer, train_loss, train_accuracy, mask_types, inputs,
                               target):
        individual_loss = strategy.run(Trainer.train_step,
                                       args=(transformer, optimizer, train_loss, train_accuracy, mask_types, inputs,
                                             target))
        overall_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, individual_loss, axis=None)

        # get metrics
        train_loss(overall_loss)

        return overall_loss
