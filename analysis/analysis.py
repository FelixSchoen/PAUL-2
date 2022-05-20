import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.config.settings import ROOT_PATH
from src.network.generator import TemperatureSchedule
from src.util.logging import get_logger
from src.util.util import pickle_load, convert_difficulty


def test_analysis():
    logger = get_logger(__name__)

    lead_difs = []
    acmp_difs = []
    lengths = []

    file_paths = []

    for dir_path, _, filenames in os.walk(f"{ROOT_PATH}/out/bars/train"):
        for file_name in [f for f in filenames if f.endswith(".zip")]:
            file_paths.append((os.path.join(dir_path, file_name), file_name))

    for file_tuple in file_paths:
        file_path, file_name = file_tuple

        logger.info(f"Loading {file_name}...")
        bars = pickle_load(file_path)

        for bar_tuple in bars:
            current_length_lead = 0
            current_length_acmp = 0
            lead_bars, acmp_bars = bar_tuple

            for bar in lead_bars:
                lead_difs.append(
                    convert_difficulty(bar.difficulty()) + 1)
                current_length_lead += len(bar._sequence._get_rel().messages)
            for bar in acmp_bars:
                acmp_difs.append(
                    convert_difficulty(bar.difficulty()) + 1)
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


def test_save_sample_bars():
    logger = get_logger(__name__)

    file_paths = []

    for dir_path, _, filenames in os.walk(f"{ROOT_PATH}/out/bars/train"):
        for file_name in [f for f in filenames if f.endswith(".zip")]:
            file_paths.append((os.path.join(dir_path, file_name), file_name))

    for file_tuple in file_paths:
        file_path, file_name = file_tuple

        logger.info(f"Loading {file_name}...")
        bars = pickle_load(file_path)

        path = f"{ROOT_PATH}/out/bars/analysis_difficulty"

        for bar_tuple in bars:
            lead_bars, acmp_bars = bar_tuple

            for bar_list in [lead_bars, acmp_bars]:
                for bar in bar_list:
                    file_path = f"{path}/{convert_difficulty(bar.difficulty())+1}_{file_name[:-4]}.mid"

                    if os.path.exists(file_path):
                        continue
                    bar._sequence.save(file_path)


def difficulty_diagram(difficulties):
    plt.hist(difficulties, bins=np.arange(12) - 0.5, color="black")

    plt.xlabel("Difficulty Class", fontsize=16, labelpad=15)
    plt.ylabel("Amount Bars", fontsize=16, labelpad=15)

    plt.xticks(range(1, 11))
    plt.xlim([0.5, 10.5])

    plt.tight_layout()

    plt.show()


def length_diagram(lengths):
    plt.hist(lengths, np.linspace(0, 512, 128), color="black")
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
