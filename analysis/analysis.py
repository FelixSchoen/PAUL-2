import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sCoda import Sequence

from src.config.settings import ROOT_PATH
from src.network.generator import TemperatureSchedule
from src.util.logging import get_logger
from src.util.util import pickle_load, convert_difficulty
from src.network.masking import create_padding_mask, create_look_ahead_mask, MaskType, create_combined_mask, \
    create_single_out_mask


def test_analysis():
    logger = get_logger(__name__)

    file_paths = []

    for dir_path, _, filenames in os.walk(f"{ROOT_PATH}/out/bars/train"):
        for file_name in [f for f in filenames if f.endswith(".zip")]:
            file_paths.append((os.path.join(dir_path, file_name), file_name))

    file_paths = file_paths

    pool = Pool()

    result = list(pool.starmap(_analyse_values, zip(file_paths)))

    logger.info("Loaded all files...")

    lead_difs = []
    acmp_difs = []
    lengths = []

    for result in result:
        lead_dif, acmp_dif, length = result

        lead_difs.extend(lead_dif)
        acmp_difs.extend(acmp_dif)
        lengths.extend(length)

    combined_difs = []
    combined_difs.extend(lead_difs)
    combined_difs.extend(acmp_difs)

    difficulty_diagram(lead_difs, "lead_difs")
    logger.info("Drawn Lead Dif Diagram...")
    difficulty_diagram(acmp_difs, "acmp_difs")
    logger.info("Drawn Acmp Dif Diagram...")
    difficulty_diagram(combined_difs, "combined_difs")
    logger.info("Drawn Combined Dif Diagram...")
    length_diagram(lengths, "lengths")
    logger.info("Drawn Length Diagram...")


def _analyse_values(file_tuple):
    logger = get_logger(__name__)

    lead_difs = []
    acmp_difs = []
    lengths = []

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
            current_length_lead += len(bar.sequence.rel.messages)

            constructed_name = f"{file_name}_{convert_difficulty(bar.difficulty()) + 1}"
            save_path = f"bars/{constructed_name}.mid"
            if not os.path.exists(save_path):
                bar.sequence.save(save_path)
        for bar in acmp_bars:
            acmp_difs.append(
                convert_difficulty(bar.difficulty()) + 1)
            current_length_acmp += len(bar.sequence.rel.messages)

        lengths.append(current_length_lead)
        lengths.append(current_length_acmp)

    return lead_difs, acmp_difs, lengths


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
                    file_path = f"{path}/{convert_difficulty(bar.difficulty()) + 1}_{file_name[:-4]}.mid"

                    if os.path.exists(file_path):
                        continue
                    bar._sequence.save(file_path)


def test_masks():
    x = tf.constant([[1, 1, 1, 0, 0]])
    print(create_combined_mask(x)[0][0])
    plt.matshow(create_combined_mask(x)[0][0], cmap='gray')
    plt.tight_layout()
    plt.xlabel("Element", fontsize=16, labelpad=15)
    plt.ylabel("Step", fontsize=16, labelpad=15)
    plt.savefig("combined.svg")

    plt.matshow(create_look_ahead_mask(5), cmap='gray')
    plt.tight_layout()
    plt.xlabel("Element", fontsize=16, labelpad=15)
    plt.ylabel("Step", fontsize=16, labelpad=15)
    plt.savefig("lookahead.svg")

    plt.matshow(create_single_out_mask(5), cmap='gray')
    plt.tight_layout()
    plt.xlabel("Element", fontsize=16, labelpad=15)
    plt.ylabel("Step", fontsize=16, labelpad=15)
    plt.savefig("singleout.svg")


def difficulty_diagram(difficulties, name_to_save):
    plt.figure()
    plt.hist(difficulties, bins=np.arange(12) - 0.5, color="black")

    plt.xlabel("Difficulty Class", fontsize=16, labelpad=15)
    plt.ylabel("Amount Bars", fontsize=16, labelpad=15)

    plt.xticks(range(1, 11))
    plt.xlim([0.5, 10.5])

    plt.tight_layout()

    plt.savefig(f"out/{name_to_save}.svg")


def length_diagram(lengths, name_to_save):
    plt.figure()
    plt.hist(lengths, np.linspace(0, 512, 128), color="black")
    plt.xlim([0, 512])
    plt.ylim(plt.ylim())

    max_length = max(lengths)
    plt.plot([max_length, max_length], plt.ylim())

    plt.xlabel("Length of Sequence", fontsize=16, labelpad=15)
    plt.ylabel("Amount Sequences", fontsize=16, labelpad=15)

    plt.tight_layout()

    plt.savefig(f"out/{name_to_save}.svg")


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
