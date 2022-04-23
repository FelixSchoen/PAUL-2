import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from src.data_processing.data_pipeline import load_stored_bars, load_dataset, load_midi_files, Detokenizer
from src.settings import DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_load_sparse_files():
    load_midi_files("D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/sparse_data")


def test_load_files():
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
