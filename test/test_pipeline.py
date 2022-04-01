import os

from src.data_processing.data_pipeline import load_stored_bars
from src.util.util_visualiser import get_message_lengths_and_difficulties

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_load_files():
    bars = load_stored_bars("D:/Documents/Coding/Repository/Badura/out/pickle")
    get_message_lengths_and_difficulties(bars)

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
