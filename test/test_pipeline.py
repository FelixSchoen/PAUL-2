import os

import tensorflow as tf

from src.data_processing.data_pipeline import load_stored_bars, undefined
from src.settings import DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_load_files():
    # store_midi_files("D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/sparse_data")
    bars = load_stored_bars(DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH)
    undefined(bars)


def test_asdf():
    lead = [[1, 2, 34], [4, 5, 6]]
    lead_diff = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    acmp = [[1, 2, 3], [4, 5, 6]]
    acmp_diff = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    ds = tf.data.Dataset.from_tensor_slices((lead, lead_diff, acmp, acmp_diff))
    print(ds)

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
