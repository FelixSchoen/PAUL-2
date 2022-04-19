import logging
import os

from src.data_processing.data_pipeline import load_stored_bars, load_dataset, load_midi_files
from src.settings import DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_load_files():
    load_midi_files("D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/sparse_data")
    bars = load_stored_bars(DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH)
    ds = load_dataset(bars)

    for batch in ds.as_numpy_iterator():
        for entry in batch:
            print("New Entry")
            lead_msg, lead_dif, acmp_msg, acmp_dif = entry
            print(lead_msg)
            break
        break


def test_asdf():
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

# TODO Cleanup
# def test_asdf():
#     composition = Composition.from_file("resources/beethoven_o27-2_m1.mid", [[1, 2], [3]], [0, 4])
#     dataframes = [bar.to_relative_dataframe() for bar in composition.tracks[0].bars]
#
#     tensors = []
#
#     for dataframe in dataframes:
#         tensors.append(dataframe_to_numeric_representation(dataframe))
#
#     dataset = tf.data.Dataset.from_tensors(tensors)
#     print(dataset)
#
#     for element in dataset:
#         print(element)
