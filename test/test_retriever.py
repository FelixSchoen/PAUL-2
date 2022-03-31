import os

import tensorflow as tf
from sCoda import Composition

from src.music.input_pipeline import store_midi_files, load_stored_bars
from src.music.retriever import dataframe_to_numeric_representation
from src.settings import DATA_PATH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_load_files():

    # load_midi_files("resources")
    store_midi_files(DATA_PATH)
    load_stored_bars("D:/Documents/Coding/Repository/Badura/out/pickle")


def test_asdf():
    composition = Composition.from_file("resources/beethoven_o27-2_m1.mid", [[1, 2], [3]], [0, 4])
    dataframes = [bar.to_relative_dataframe() for bar in composition.tracks[0].bars]

    tensors = []

    for dataframe in dataframes:
        tensors.append(dataframe_to_numeric_representation(dataframe))

    dataset = tf.data.Dataset.from_tensors(tensors)
    print(dataset)

    for element in dataset:
        print(element)
