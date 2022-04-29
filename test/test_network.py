import time

import tensorflow as tf
import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import tensorflow_text
from matplotlib import pyplot as plt

from src.network.attention import scaled_dot_product_attention, AttentionType, skew, \
    relative_scaled_dot_product_attention
from src.network.layers import MultiHeadAttention, PointwiseFeedForwardNetwork, EncoderLayer, DecoderLayer
from src.network.masking import create_padding_mask, create_look_ahead_mask, MaskType, create_combined_mask, \
    create_single_out_mask
from src.network.optimization import TransformerLearningRateSchedule
from src.network.positional_encoding import positional_encoding
from src.network.training import Trainer
from src.network.transformer import Encoder, Decoder, Transformer
from src.settings import D_MODEL
from src.util.logging import get_logger

logger = get_logger(__name__)


def test_positional_encoding():
    n, d = 2048, 512
    pos_encoding = positional_encoding(n, d)

    logger.info(f"Shape: {pos_encoding.shape}")
    logger.info(f"Encoding: {pos_encoding}")

    pos_encoding = pos_encoding[0]

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Position')
    plt.xlabel('Depth')
    plt.colorbar()
    plt.show()


def test_padding_mask():
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    logger.info(f"Padding mask: {create_padding_mask(x)}")


def test_look_ahead_mask():
    x = tf.random.uniform((1, 3))
    logger.info(f"Look ahead mask: {create_look_ahead_mask(x.shape[1])}")


def test_single_out_mask():
    x = tf.random.uniform((1, 3))
    logger.info(f"Single out mask: {create_single_out_mask(x.shape[1])}")
    plt.matshow(create_single_out_mask(5))
    plt.show()


def test_scaled_dot_product_attention():
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    logger.info(f"Attention: {temp_attn}")
    logger.info(f"Output: {temp_out}")

    # This query aligns with a repeated key (third and fourth),
    # so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)

    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    logger.info(f"Attention: {temp_attn}")
    logger.info(f"Output: {temp_out}")


def test_split_heads():
    t = tf.random.normal((64, 10, 200))

    mha = MultiHeadAttention(num_heads=8, d_model=200, attention_type=AttentionType.absolute)

    logger.info(f"Split: {tf.shape(mha.split_heads(t))}")


def test_multi_head_attention():
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8, attention_type=AttentionType.absolute)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(q=y, k=y, v=y, mask=None)

    logger.info(f"Output shape: {out.shape}")
    logger.info(f"Attention shape: {attn.shape}")


def test_pointwise_feed_forward_network():
    sample_ffn = PointwiseFeedForwardNetwork(d_model=512, dff=2048)
    shape = sample_ffn(tf.random.uniform((64, 50, 512))).shape

    logger.info(f"Shape: {shape}")


def test_encoder_layer():
    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048, attention_type=AttentionType.absolute,
                                        max_relative_distance=None)

    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    logger.info(f"Shape: {sample_encoder_layer_output.shape}")

    return sample_encoder_layer_output


def test_decoder_layer():
    sample_encoder_layer_output = test_encoder_layer()

    sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048, num_encoders=1,
                                        attention_type=AttentionType.absolute, max_relative_distance=None)

    sample_decoder_layer_output, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)),
                                                          [sample_encoder_layer_output], False, None, [None])

    logger.info(f"Shape: {sample_decoder_layer_output.shape}")


def test_encoder():
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500,
                             attention_type=AttentionType.absolute, max_relative_distance=None)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    logger.info(f"Shape: {sample_encoder_output.shape}")

    return sample_encoder_output


def test_decoder():
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000, num_encoders=1, attention_type=AttentionType.absolute,
                             max_relative_distance=None)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_outputs=[test_encoder()],
                                  training=False,
                                  self_attention_mask=None,
                                  enc_masks=[None])

    logger.info(f"Output Shape: {output.shape}")
    logger.info(f"Attention Shape: {attn['decoder_layer2_block2'].shape}")


def test_transformer():
    sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_sizes=[8500],
                                     target_vocab_size=8000, num_encoders=1, attention_type=AttentionType.absolute,
                                     max_relative_distance=None)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer([[temp_input], temp_target], training=False, mask_types=[MaskType.padding])

    logger.info(f"Shape: {fn_out.shape}")


def test_learning_rate():
    temp_learning_rate_schedule = TransformerLearningRateSchedule(D_MODEL)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel('Learning Rate')
    plt.xlabel('Train Step')


def test_skew():
    u = tf.constant([[0, 1, 1, 0, 2],
                     [1, 0, 0, 3, 2],
                     [1, 1, 5, 3, 2],
                     [0, 7, 5, 3, 2],
                     [9, 7, 5, 3, 2]], dtype=tf.float32)

    plots = [u, skew(u)]

    fig = plt.figure(figsize=(10, 6.5))
    rows = 1
    cols = 2
    labels = ['u', 'skew(u)']

    for i in range(rows * cols):
        fig.add_subplot(1, 2, i + 1).set_title(labels[i], fontsize=14)
        plt.imshow(plots[i], cmap='viridis')
    fig.tight_layout()
    plt.show()


def test_relative_scaled_dot_product_attention():
    temp_k = tf.constant([[0, 0, 10], [0, 10, 0], [10, 0, 0], [10, 0, 0]], dtype=tf.float32)
    temp_v = tf.constant([[4, 2, 1], [5, 6, 3], [7, 8, 10], [9, 12, 45]], dtype=tf.float32)
    temp_e = tf.zeros_like(temp_k)  # zero the relative position embeddings to demonstrate original attention

    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)
    attn, attn_weights = relative_scaled_dot_product_attention(temp_q, temp_k, temp_v, temp_e, None)

    logger.info(f"Attention: {attn}")
    logger.info(f"Weights: {attn_weights}")

    # Relative embeddings change outcome:
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # aligns with second key

    temp_e = tf.constant([[-1, -1, -10], [2, 2, 2], [1, 1, 1], [4, 4, 4]], dtype=tf.float32)

    attn, attn_weights = relative_scaled_dot_product_attention(temp_q, temp_k, temp_v, temp_e, None)

    logger.info(f"Attention: {attn}")
    logger.info(f"Weights: {attn_weights}")


def test_embeddings():
    E = tf.keras.layers.Embedding(400, 200)
    logger.info(f"Shape: {MultiHeadAttention.get_embeddings(E, 500, None).shape}")


def test_relative_multi_head_attention():
    # Create a MultiHeadAttention Block to test
    t = tf.random.uniform((10, 1500, 256))
    mha = MultiHeadAttention(d_model=256, num_heads=8, attention_type=AttentionType.relative,
                             max_relative_distance=1921)
    out, attn = mha(t, t, t, create_combined_mask(tf.random.uniform((10, 1500))))

    logger.info(f"Output Shape: {out.shape}")
    logger.info(f"Attention Shape: {attn.shape}")
    logger.info(f"Trainable variables: {len(mha.trainable_variables)}")


def test_combined():
    tokenizers, train_batches, val_batches, max_tokens = _get_demo_dataset()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.Mean(name="val_accuracy")

    learning_rate = TransformerLearningRateSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    transformer = Transformer(num_layers=4,
                              d_model=128,
                              num_heads=8,
                              dff=512,
                              input_vocab_sizes=[tokenizers.pt.get_vocab_size().numpy(),
                                                 tokenizers.pt.get_vocab_size().numpy()],
                              target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
                              num_encoders=2,
                              attention_type=AttentionType.absolute,
                              max_relative_distance=max_tokens)

    epochs = 1

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        trainer = Trainer(transformer, optimizer, train_loss, train_accuracy, val_loss, val_accuracy,
                          [MaskType.padding, MaskType.padding], strategy=None)

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            trainer.train_step([inp, inp], tar)

            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


def _get_demo_dataset():
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
