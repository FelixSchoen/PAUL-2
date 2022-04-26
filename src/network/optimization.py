from contextlib import nullcontext

import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_loss_object(strategy=None):
    if strategy is None:
        context = nullcontext()
    else:
        context = strategy.scope()

    with context:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction=tf.keras.losses.Reduction.NONE)
        return loss_object


def loss_function(real, pred, strategy=None):
    if strategy is None:
        context = nullcontext()
    else:
        context = strategy.scope()

    with context:
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        _loss = get_loss_object(strategy)(real, pred)

        mask = tf.cast(mask, dtype=_loss.dtype)
        _loss *= mask

        return tf.reduce_sum(_loss) / tf.reduce_sum(mask)


def accuracy_function(real, pred, strategy=None):
    if strategy is None:
        context = nullcontext()
    else:
        context = strategy.scope()

    with context:
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=tf.int16))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
