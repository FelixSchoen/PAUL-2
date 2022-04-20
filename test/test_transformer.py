import time

import tensorflow as tf

from src.data_processing.data_pipeline import load_stored_bars, load_dataset
from src.network.masking import MaskType
from src.network.optimization import TransformerSchedule
from src.network.training import Trainer
from src.network.transformer import Transformer
from src.settings import D_MODEL, NUM_HEADS, DFF, NUM_LAYERS, INPUT_VOCAB_SIZE_DIF, OUTPUT_VOCAB_SIZE, DROPOUT_RATE, \
    DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH, PATH_CHECKPOINT_LEAD, INPUT_VOCAB_SIZE_MLD


def test_transformer():
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        h=NUM_HEADS,
        dff=DFF,
        num_encoders=2,
        input_vocab_sizes=[INPUT_VOCAB_SIZE_MLD, INPUT_VOCAB_SIZE_DIF],
        target_vocab_size=OUTPUT_VOCAB_SIZE,
        rate=DROPOUT_RATE,
        attention_type="relative")

    bars = load_stored_bars(DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH)
    ds = load_dataset(bars)

    learning_rate = TransformerSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    checkpoint_path = PATH_CHECKPOINT_LEAD

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    EPOCHS = 1

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch_num, entry) in enumerate(ds.as_numpy_iterator()):
            lead_seqs, lead_difs, acmp_seqs, acmp_difs = [], [], [], []
            for lead_seq, lead_dif, acmp_seq, acmp_dif in entry:
                lead_seqs.append(lead_seq)
                lead_difs.append(lead_dif)
                acmp_seqs.append(acmp_seq)
                acmp_difs.append(acmp_dif)

            lead_seq = tf.stack(lead_seqs)
            acmp_seq = tf.stack(acmp_seqs)
            acmp_dif = tf.stack(acmp_difs)

            trainer = Trainer(transformer, optimizer, train_loss, train_accuracy)

            trainer([lead_seq, acmp_dif], acmp_seq, [MaskType.padding, MaskType.lookahead])

            if batch_num % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch_num} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        # if (epoch + 1) % 5 == 0:
        #     ckpt_save_path = ckpt_manager.save()
        #     print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
