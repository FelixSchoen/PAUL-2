import tensorflow as tf

from src.exception.exceptions import UnexpectedValueException
from src.network.masking import create_padding_mask, create_combined_mask, MaskType
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
        self.train_step(self.transformer, self.optimizer, self.train_loss, self.train_accuracy, args[2], args[0],
                        args[1])

    @staticmethod
    @tf.function
    def train_step(transformer, optimizer, train_loss, train_accuracy, mask_types, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_masks = []
        dec_masks = [create_combined_mask(tar_inp)]

        # Create masks
        for i, s_inp in enumerate(inp):
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
            predictions, _ = transformer([inp, tar_inp], enc_masks, dec_masks, training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))
