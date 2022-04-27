import tensorflow as tf
import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import tensorflow_text


def get_demo_dataset():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir="./resources/", extract=True
    )

    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    MAX_TOKENS = 128

    tokenizers = tf.saved_model.load("./resources/datasets/" + model_name)

    def filter_max_tokens(pt, en):
        num_tokens = tf.maximum(tf.shape(pt)[1], tf.shape(en)[1])
        return num_tokens < MAX_TOKENS

    def tokenize_pairs(pt, en):
        pt = tokenizers.pt.tokenize(pt)
        # Convert from ragged to dense, padding with zeros.
        pt = pt.to_tensor()

        en = tokenizers.en.tokenize(en)
        # Convert from ragged to dense, padding with zeros.
        en = en.to_tensor()
        return pt, en

    def make_batches(ds):
        return (
            ds
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
                .filter(filter_max_tokens)
                .prefetch(tf.data.AUTOTUNE))

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    return tokenizers, train_batches, val_batches, MAX_TOKENS
