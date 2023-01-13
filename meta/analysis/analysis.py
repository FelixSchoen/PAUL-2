import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sCoda import Sequence, Message, Bar
from sCoda.elements.message import MessageType

from src.config.settings import ROOT_PATH, PATH_SAVED_MODEL, MAXIMUM_NOTE_LENGTH, DATA_MIDI_INPUT_PATH, \
    DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH, DATA_BARS_VAL_OUTPUT_FOLDER_PATH
from src.network.generator import TemperatureSchedule, Generator
from src.network.masking import create_look_ahead_mask, create_combined_mask, \
    create_single_out_mask
from src.network.paul import get_network_objects
from src.util.enumerations import NetworkType
from src.util.logging import get_logger
from src.util.util import pickle_load, convert_difficulty


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


def test_analyse_confusion_matrix():
    # actual_difficulties = []
    # output_difficulties = []
    #
    # for dif in range(0, 10):
    #     for _ in range(5):
    #         new_difs = generate(NetworkType.lead, "lead", dif)
    #         actual_difficulties.extend(dif for _ in range(len(new_difs)))
    #         output_difficulties.extend(new_difs)
    #
    # data = {"actual": actual_difficulties, "output": output_difficulties}
    # df = pd.DataFrame(data)
    # df.to_pickle("out/dataframe.pkl")

    sequences = Sequence.from_midi_file("res/chopin_o66_fantaisie_impromptu.mid", [[1], [2]], [0])
    for sequence in sequences:
        sequence.quantise()
        sequence.quantise_note_lengths()

    sequence_lead = sequences[0]

    # Split sequence into bars
    bar = Sequence.split_into_bars([sequence_lead])[0][4]
    seq = Bar.to_sequence([bar])

    actual_difficulties = []
    output_difficulties = []

    for dif in range(0, 10):
        for _ in range(5):
            new_difs = generate(NetworkType.acmp, "acmp", dif, lead_seq=seq)
            actual_difficulties.extend(dif for _ in range(len(new_difs)))
            output_difficulties.extend(new_difs)

    data = {"actual": actual_difficulties, "output": output_difficulties}
    df = pd.DataFrame(data)
    df.to_pickle("out/dataframe.pkl")


def test_search_real_pieces_of_difficulties():
    difficulties = [1, 4, 7]

    file_paths = []
    for dir_path, _, filenames in os.walk(DATA_BARS_VAL_OUTPUT_FOLDER_PATH):
        for file_name in [f for f in filenames if f.endswith(".zip")]:
            file_paths.append((os.path.join(dir_path, file_name), file_name))

    found_index = 0

    for file_path in file_paths:
        bars = pickle_load(file_path[0])
        for group in bars:
            lead = group[0]
            acmp = group[1]

            lead_dif_min = min([convert_difficulty(bar.difficulty()) for bar in lead])
            lead_dif_max = max([convert_difficulty(bar.difficulty()) for bar in lead])
            acmp_dif_min = min([convert_difficulty(bar.difficulty()) for bar in acmp])
            acmp_dif_max = max([convert_difficulty(bar.difficulty()) for bar in acmp])

            lead_seq = Bar.to_sequence(lead)
            acmp_seq = Bar.to_sequence(acmp)

            for msg in lead_seq.rel.messages:
                if msg.message_type == MessageType.note_on:
                    msg.velocity = 127
            for msg in acmp_seq.rel.messages:
                if msg.message_type == MessageType.note_on:
                    msg.velocity = 127

            time_sig = [msg for msg in lead_seq.rel.messages if msg.message_type == MessageType.time_signature]
            if not all(msg.numerator == 4 and msg.denominator == 4 for msg in time_sig):
                continue

            # Lead
            for dif in difficulties:
                if lead_dif_min >= dif - 1 and lead_dif_max <= dif + 1:
                    lead_seq.save(f"out/survey_samples/{found_index:06d}-lead-d{dif}-{file_path[1]}.mid")
                    found_index += 1
                    break

            # Acmp
            for dif in difficulties:
                if acmp_dif_min >= dif - 1 and acmp_dif_max <= dif + 1:
                    acmp_seq.save(f"out/survey_samples/{found_index:06d}-acmp-d{dif}-{file_path[1]}.mid")
                    lead_seq.save(f"out/survey_samples/{found_index:06d}-acmpL-d{dif}-{file_path[1]}.mid")
                    found_index += 1
                    break


def generate(network_type: NetworkType, model_identifier: str, difficulty: int,
             primer_sequence: Sequence = None, lead_seq: Sequence = None):
    logger = get_logger(__name__)

    assert not network_type == NetworkType.acmp or lead_seq is not None

    # Load model
    logger.info("Constructing model...")
    transformer, _ = get_network_objects(network_type)
    transformer.build_model()
    transformer.load_weights(f"{PATH_SAVED_MODEL}/{network_type.value}/model_{model_identifier}.h5")

    # Get difficulties of generated bars
    output_difficulties = []

    # Create sequence object
    gen_seq = primer_sequence if primer_sequence is not None else Sequence()

    # Start with 4/4 time signature if not provided
    if primer_sequence is None:
        gen_seq.rel.messages.append(Message(message_type=MessageType.time_signature, numerator=4, denominator=4))

    # Load generator
    generator = Generator(transformer, network_type, lead_sequence=lead_seq)

    sequences, attention_weights = generator(input_sequence=gen_seq, difficulty=difficulty,
                                             temperature=0.6,
                                             bars_to_generate=1)

    for i, sequence in enumerate(sequences):
        print(len(Sequence.split_into_bars([sequence])[0]))

        sequence.quantise()
        sequence.quantise_note_lengths()
        sequence.cutoff(2 * MAXIMUM_NOTE_LENGTH, MAXIMUM_NOTE_LENGTH)

        bar = Sequence.split_into_bars([sequence])[0][0]
        output_difficulties.append(convert_difficulty(bar.difficulty()))

    return output_difficulties


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
