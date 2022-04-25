import time
from logging import getLogger
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from src.data_processing.data_pipeline import load_stored_bars, load_dataset, load_midi_files, Detokenizer, \
    load_oom_dataset, _bar_tuple_to_token_tuple, _filter_length, _pad_tuples
from src.settings import DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH


def test_load_sparse_midi_files():
    load_midi_files("D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/sparse_data")


def test_pipeline():
    load_midi_files("D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/sparse_data")
    bars = load_stored_bars(DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH)
    ds = load_dataset(bars)

    for batch in ds.as_numpy_iterator():
        for entry in batch:
            print("New Entry")
            lead_msg, lead_dif, acmp_msg, acmp_dif = entry
            print(lead_msg)

            data = []

            detokenizer = Detokenizer()

            for x in lead_msg:
                data.extend(detokenizer.detokenize(x))
            data.extend(detokenizer.flush_wait_buffer())

            df = pd.DataFrame(data)
            print(df)

            break
        break


def test_count_signatures():
    files = load_midi_files("D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/data",
                            flags=["skip_difficulty", "skip_store", "skip_skip"])

    signatures = dict()

    for composition in files:
        for bars in composition:
            lead_track_bars = bars[0]
            for bar in lead_track_bars:
                df = bar.to_relative_dataframe()

                for _, row in df.iterrows():
                    msg_type = row["message_type"]

                    if msg_type != "time_signature":
                        continue

                    num = int(row["numerator"])
                    den = int(row["denominator"])
                    tpl = (num, den)

                    signatures[tpl] = signatures.get(tpl, 0) + 1

    print(signatures)
    print(sorted(signatures, key=signatures.get))


def test_count_length():
    bars = load_stored_bars(DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH)
    ds = load_dataset(bars)

    a = []

    for batch in ds.as_numpy_iterator():
        for entry in batch:
            lead_msg, _, acmp_msg, _ = entry

            a.append(tf.math.count_nonzero(lead_msg).numpy())
            a.append(tf.math.count_nonzero(acmp_msg).numpy())

    max_length = max(a)

    plt.hist(a, np.linspace(1, 512, 512))
    plt.ylim(plt.ylim())
    plt.plot([max_length, max_length], plt.ylim())
    plt.show()


def test_time_load():
    start_time = time.perf_counter()
    bars = load_stored_bars("D:/Documents/Coding/Repository/Badura/out/pickle_sparse/compositions")
    end_time = time.perf_counter()

    print(f"Time needed for loading bars: {end_time - start_time}")

    start_time = time.perf_counter()
    ds = load_dataset(bars)
    end_time = time.perf_counter()

    print(f"Time needed for loading dataset: {end_time - start_time}")


def test_custom_generator():
    logger = getLogger("badura." + __name__)

    start_time = time.perf_counter()
    logger.info("Loading dataset")
    ds = load_oom_dataset(directory="D:/Documents/Coding/Repository/Badura/out/pickle_sparse/compositions")

    i = 0
    e = 0
    for batch in ds.as_numpy_iterator():
        i += 1
        for entry in batch:
            e += 1
            if e == 1:
                lead_msg, lead_dif, acmp_msg, acmp_dif = entry

    end_time = time.perf_counter()

    logger.info(f"{i} Batches, {e} Entries")

    logger.info(f"Time needed for loading dataset: {end_time - start_time}")


def test_save_dataset_to_file():
    logger = getLogger("badura." + __name__)

    logger.info("Loading dataset")
    ds = load_oom_dataset()

    path = "D:/Documents/Coding/Repository/Badura/out/dataset/"

    logger.info("Saving dataset")
    tf.data.experimental.save(ds, path)

    logger.info("Loading dataset")
    new_dataset = tf.data.experimental.load(path, element_spec=(
        tf.TensorSpec(shape=(None, 4, 512), dtype=tf.int16)
    ))

    logger.info("Loaded dataset")

    for batch in new_dataset.as_numpy_iterator():
        for entry in batch:
            print("New Entry")
            lead_msg, lead_dif, acmp_msg, acmp_dif = entry
            print(lead_msg)

    print(ds.element_spec)


def test_compare_speed_oom_normal():
    logger = getLogger("badura." + __name__)

    start_time = time.perf_counter()
    bars = load_stored_bars("D:/Documents/Coding/Repository/Badura/out/pickle_sparse/compositions")
    ds = load_dataset(bars)
    list(ds)
    del bars, ds
    end_time = time.perf_counter()
    logger.info(f"Time needed for loading dataset: {end_time - start_time}")

    start_time = time.perf_counter()
    ds = load_oom_dataset("D:/Documents/Coding/Repository/Badura/out/pickle_sparse/compositions")
    list(ds)
    end_time = time.perf_counter()
    logger.info(f"Time needed for loading dataset: {end_time - start_time}")


def tensor_to_record(lead_msg, lead_dif, acmp_msg, acmp_dif):
    def _int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    feature = {
        "lead_msg": _int_feature(lead_msg),
        "lead_dif": _int_feature(lead_dif),
        "acmp_msg": _int_feature(acmp_msg),
        "acmp_dif": _int_feature(acmp_dif),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _serialize_example(entry):
    lead_msg, lead_dif, acmp_msg, acmp_dif = entry
    record = tensor_to_record(lead_msg, lead_dif, acmp_msg, acmp_dif)
    return record.SerializeToString()


def test_convert_to_tfrecords():
    bars = load_stored_bars("D:/Documents/Coding/Repository/Badura/out/pickle_sparse/compositions")
    data_rows = map(_bar_tuple_to_token_tuple, bars)
    data_rows = filter(_filter_length, data_rows)
    data_rows = map(_pad_tuples, data_rows)

    pool = Pool()
    with tf.io.TFRecordWriter("D:/Documents/Coding/Repository/Badura/out/dataset/test.tfrecords") as writer:
        for example in pool.map(_serialize_example, list(data_rows)):
            print("Writing")
            writer.write(example)


def test_read_tfrecords():
    files = ["D:/Documents/Coding/Repository/Badura/out/dataset/test.tfrecords"]

    raw_dataset = tf.data.TFRecordDataset(files)

    feature_desc = {
        "lead_msg": tf.io.FixedLenFeature([512], tf.int64),
        "lead_dif": tf.io.FixedLenFeature([512], tf.int64),
        "acmp_msg": tf.io.FixedLenFeature([512], tf.int64),
        "acmp_dif": tf.io.FixedLenFeature([512], tf.int64),
    }

    def _parse_function(example_proto):
        dictionary = tf.io.parse_single_example(example_proto, feature_desc)
        return tf.stack(
            [tf.cast(dictionary["lead_msg"], dtype=tf.int16),
             tf.cast(dictionary["lead_msg"], dtype=tf.int16),
             tf.cast(dictionary["lead_msg"], dtype=tf.int16),
             tf.cast(dictionary["lead_msg"], dtype=tf.int16),
             ])

    ds = raw_dataset.map(_parse_function).batch(16)

    print(ds)

    for batch in ds.as_numpy_iterator():
        print(f"Batch length: {len(batch)}")
        for entry in batch:
            a, b, c, d = entry
            print(tf.shape(tf.cast(a, dtype=tf.int16)))
            break
        print(tf.shape(batch))
        break
