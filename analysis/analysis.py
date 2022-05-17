import math

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.config.settings import ROOT_PATH, DIFFICULTY_VALUE_SCALE
from src.network.generator import TemperatureSchedule
from src.preprocessing.data_pipeline import _load_bars_from_zip


def test_analysis():
    # bars = load_stored_bars(directory=DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH)
    bars = _load_bars_from_zip(directory=ROOT_PATH + "/out/bars/misc")

    lead_difs = []
    acmp_difs = []
    lengths = []

    for bar_tuple in bars:
        current_length_lead = 0
        current_length_acmp = 0
        lead_bars, acmp_bars = bar_tuple

        for bar in lead_bars:
            lead_difs.append(math.floor(min(DIFFICULTY_VALUE_SCALE - 1, bar.difficulty() * DIFFICULTY_VALUE_SCALE)))
            current_length_lead += len(bar._sequence._get_rel().messages)
        for bar in acmp_bars:
            acmp_difs.append(math.floor(min(DIFFICULTY_VALUE_SCALE - 1, bar.difficulty() * DIFFICULTY_VALUE_SCALE)))
            current_length_acmp += len(bar._sequence._get_rel().messages)

        lengths.append(current_length_lead)
        lengths.append(current_length_acmp)

    combined_difs = []
    combined_difs.extend(lead_difs)
    combined_difs.extend(acmp_difs)

    difficulty_diagram(lead_difs)
    difficulty_diagram(acmp_difs)
    difficulty_diagram(combined_difs)
    length_diagram(lengths)


def difficulty_diagram(difficulties):
    plt.hist(difficulties, bins=np.arange(10) - 0.5, color="black")

    plt.xlabel("Difficulty Class", fontsize=16, labelpad=15)
    plt.ylabel("Amount Bars", fontsize=16, labelpad=15)

    plt.xticks(range(1, 11))
    plt.xlim([0.5, 10.5])

    plt.tight_layout()

    plt.show()


def length_diagram(lengths):
    plt.hist(lengths, np.linspace(0, 512, 513), color="black")
    plt.xlim([0, 512])
    plt.ylim(plt.ylim())

    max_length = max(lengths)
    plt.plot([max_length, max_length], plt.ylim())

    plt.xlabel("Length of Sequence", fontsize=16, labelpad=15)
    plt.ylabel("Amount Sequences", fontsize=16, labelpad=15)

    plt.tight_layout()

    plt.show()


def test_temperature_rate():
    temp = TemperatureSchedule(96, 12, 1 / 2, exponent=2.5, max_value=1, min_value=0.2)

    plt.plot(temp(tf.range(100, dtype=tf.float32)), color="black")

    plt.xticks([0, 24, 48, 72, 96, 120])

    plt.xlim([0, 100])
    plt.ylim([0, 1])

    plt.xlabel("Length of Sequence in Ticks", fontsize=16, labelpad=15)
    plt.ylabel("Temperature", fontsize=16, labelpad=15)
    plt.tight_layout()
    plt.show()
