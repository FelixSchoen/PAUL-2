import copy
import math
import os.path
import random
import re
from multiprocessing import Pool

import mido
import numpy as np
import tensorflow as tf
from sCoda import Composition, Bar, Sequence
from sCoda.elements.message import MessageType

from src.config.settings import SEQUENCE_MAX_LENGTH, CONSECUTIVE_BAR_MAX_LENGTH, \
    VALID_TIME_SIGNATURES, DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH, \
    START_TOKEN, STOP_TOKEN, D_TYPE, DIFFICULTY_VALUE_SCALE, TRAIN_VAL_SPLIT, \
    DATA_BARS_VAL_OUTPUT_FOLDER_PATH, SHUFFLE_SEED
from src.exception.exceptions import UnexpectedValueException
from src.util.logging import get_logger
from src.util.util import flatten, pickle_save, pickle_load


# =================
# === Load MIDI ===
# =================

def load_midi(directory: str, flags=None) -> None:
    """ Loads the MIDI files from the drive, processes them, and stores the processed files.

    Applies a train / validation split according to the percentage given in the settings, and stores the processed bars
    in different directories regarding their split.

    Args:
        directory: The directory to load the files from
        flags: A list of flags for the processing of files

    Returns: The loaded files

    """
    logger = get_logger(__name__)

    if flags is None:
        flags = []

    file_paths = []

    # Handle all MIDI files in the given directory and subdirectories
    for dir_path, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            file_paths.append(os.path.join(dir_path, filename))

    pool = Pool()

    logger.info("Loading compositions...")
    compositions = flatten(list(pool.starmap(_load_midi_load_composition_and_stretch, zip(file_paths))))

    logger.info("Loading bars...")
    bars = flatten(list(pool.starmap(_load_midi_extract_bars, zip(compositions))))

    logger.info("Calculating difficulty...")
    list(pool.starmap(_load_midi_calculate_difficulty, zip(bars)))

    # Shuffle bars
    random.Random(SHUFFLE_SEED).shuffle(bars)

    # Split into train and val data
    split_point = int((len(bars) + 1) * TRAIN_VAL_SPLIT)
    train_bars = bars[:split_point]
    val_bars = bars[split_point:]

    logger.info("Transposing bars...")
    train_bars_trans = flatten(list(map(_load_midi_transpose_bars, train_bars)))
    val_bars_trans = flatten(list(map(_load_midi_transpose_bars, val_bars)))

    logger.info("Storing bars...")
    train_zip_file_path = (DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH + "/train.zip")
    val_zip_file_path = (DATA_BARS_VAL_OUTPUT_FOLDER_PATH + "/val.zip")
    pickle_save(train_bars_trans, train_zip_file_path)
    pickle_save(val_bars_trans, val_zip_file_path)


def _load_midi_load_composition_and_stretch(file_path: str) -> [Composition]:
    """ Loads the MIDI file stored at the file path, stretches it, and returns a list of compositions.

    Args:
        file_path: File path of the MIDI file

    Returns: A list of stretched compositions

    """
    lead_tracks = []
    acmp_tracks = []
    meta_tracks = [0]

    # Open MIDI file
    midi_file = mido.MidiFile(file_path)

    # Parse track titles
    for t, track in enumerate(midi_file.tracks):
        if _load_midi_find_word("right", track.name) is not None:
            lead_tracks.append(t)
        elif _load_midi_find_word("left", track.name) is not None:
            acmp_tracks.append(t)
        elif _load_midi_find_word("pedal", track.name) is not None:
            meta_tracks.append(t)

    compositions = []
    stretch_factors = [0.5, 1, 2]

    # Load sequences from file
    sequences = Sequence.sequences_from_midi_file(file_path, [lead_tracks, acmp_tracks], meta_tracks)

    # Stretch sequences by given factors
    for stretch_factor in stretch_factors:
        stretched_sequences = []

        for sequence in sequences:
            stretched_sequence = copy.copy(sequence)
            stretched_sequence.stretch(stretch_factor)
            stretched_sequence.quantise_note_lengths()
            stretched_sequences.append(stretched_sequence)

        # Create composition from stretched sequences
        compositions.append(Composition.from_sequences(stretched_sequences))

    return compositions


def _load_midi_extract_bars(composition) -> [([Bar], [Bar])]:
    lead_track = composition.tracks[0]
    acmp_track = composition.tracks[1]

    # Check that we start with the same number of bars
    assert len(lead_track.bars) == len(acmp_track.bars)

    # Chunk the bars
    lead_chunked = []
    acmp_chunked = []

    lead_current = []
    acmp_current = []

    remaining = CONSECUTIVE_BAR_MAX_LENGTH
    for lead_bar, acmp_bar in zip(lead_track.bars, acmp_track.bars):
        # Check if time signature of bar is valid
        signature = (int(lead_bar._time_signature_numerator), int(lead_bar._time_signature_denominator))

        # Split at non-valid time signature
        if signature not in VALID_TIME_SIGNATURES:
            remaining = CONSECUTIVE_BAR_MAX_LENGTH

            if len(lead_current) > 0:
                lead_chunked.append(lead_current)
                acmp_chunked.append(acmp_current)

            lead_current = []
            acmp_current = []

            continue

        lead_current.append(lead_bar)
        acmp_current.append(acmp_bar)
        remaining -= 1

        # Maximum length reached
        if remaining == 0:
            remaining = CONSECUTIVE_BAR_MAX_LENGTH

            lead_chunked.append(lead_current)
            acmp_chunked.append(acmp_current)

            lead_current = []
            acmp_current = []

    if len(lead_current) > 0:
        lead_chunked.append(lead_current)
        acmp_chunked.append(acmp_current)

    # Zip the chunked bars
    zipped_chunks = list(zip(lead_chunked, acmp_chunked))

    return zipped_chunks


def _load_midi_calculate_difficulty(bar_tuple: ([Bar], [Bar])) -> None:
    for bar in bar_tuple[0]:
        bar.difficulty()
    for bar in bar_tuple[1]:
        bar.difficulty()

    return bar_tuple


def _load_midi_transpose_bars(base_bars: ([Bar], [Bar])):
    transposed_bars = []

    for transpose_by in range(-5, 7):
        lead_bars = []
        acmp_bars = []

        # Copy the original bar in order to transpose it later on
        lead_unedited = [copy.copy(bar) for bar in base_bars[0]]
        acmp_unedited = [copy.copy(bar) for bar in base_bars[1]]

        # Handle bars at the same time
        for lead_bar, acmp_bar in zip(lead_unedited, acmp_unedited):
            lead_bar.difficulty()
            acmp_bar.difficulty()

            # Append transposed bars to the placeholder objects
            lead_bars.append(lead_bar)
            acmp_bars.append(acmp_bar)

        transposed_bars.append((lead_bars, acmp_bars))

    return transposed_bars


def _load_midi_find_word(word, sentence) -> re.Match:
    """ Tries to find a word, not just a sequence of characters, in the given sentence

    Args:
        word: The word to look for
        sentence: The sentence in which the word may occur

    Returns: A `match` object

    """
    return re.compile(fr"\b({word})\b", flags=re.IGNORECASE).search(sentence)


# =====================
# === Store Records ===
# =====================

def store_records(input_dir, output_path):
    logger = get_logger(__name__)

    logger.info("Loading bars...")
    bars = []
    for dir_path, _, filenames in os.walk(input_dir):
        for filename in [f for f in filenames if f.endswith(".zip")]:
            bars.extend(pickle_load((os.path.join(dir_path, filename))))

    logger.info("Converting bars...")
    data_rows = map(_store_records_bar_to_tensor, bars)

    logger.info("Filtering bars...")
    data_rows = filter(_store_records_filter_length, data_rows)

    logger.info("Padding bars...")
    data_rows = map(_store_records_pad_tensors, data_rows)

    logger.info("Writing records...")
    pool = Pool()
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for example in pool.map(_store_records_serialize_tensors, list(data_rows)):
            writer.write(example)


def _store_records_bar_to_tensor(bars: ([Bar], [Bar])):
    lead_seq, lead_dif, acmp_seq, acmp_dif = [], [], [], []

    for i, (seq, dif) in enumerate([(lead_seq, lead_dif), (acmp_seq, acmp_dif)]):
        tokenizer = Tokenizer(skip_time_signature=(i == 1))

        for bar in bars[i]:
            data_frame = bar.to_relative_dataframe()

            # Sanity check
            assert len(data_frame) > 0

            # Pandas dataframe to list of tokens
            for _, row in data_frame.iterrows():
                try:
                    tokens = tokenizer.tokenize(row)
                    seq.extend(tokens)
                except UnexpectedValueException:
                    pass

            # Append trailing wait messages
            tokens = tokenizer.flush_wait_buffer()

            seq.extend(tokens)
            dif.extend([tokenizer.tokenize_difficulty(bar.difficulty()) for _ in range(0, len(seq))])

        # Add start and stop messages
        seq.insert(0, START_TOKEN)
        dif.insert(0, START_TOKEN)
        seq.append(STOP_TOKEN)
        dif.append(STOP_TOKEN)

    return tf.convert_to_tensor(lead_seq, dtype=D_TYPE), tf.convert_to_tensor(lead_dif, dtype=D_TYPE), \
           tf.convert_to_tensor(acmp_seq, dtype=D_TYPE), tf.convert_to_tensor(acmp_dif, dtype=D_TYPE)


def _store_records_filter_length(to_filter):
    length = max([tf.shape(x)[0] for x in to_filter])
    return length <= SEQUENCE_MAX_LENGTH - 2  # Leave space for start and stop messages


def _store_records_pad_tensors(tensors_to_pad):
    results = []

    for ele in tensors_to_pad:
        results.append(np.pad(ele, (0, SEQUENCE_MAX_LENGTH - ele.shape[0]), "constant"))

    return results


def _store_records_serialize_tensors(entry):
    lead_msg, lead_dif, acmp_msg, acmp_dif = entry
    record = _store_records_tensor_to_record(lead_msg, lead_dif, acmp_msg, acmp_dif)
    return record.SerializeToString()


def _store_records_tensor_to_record(lead_msg, lead_dif, acmp_msg, acmp_dif):
    def _int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    feature = {
        "lead_msg": _int_feature(lead_msg),
        "lead_dif": _int_feature(lead_dif),
        "acmp_msg": _int_feature(acmp_msg),
        "acmp_dif": _int_feature(acmp_dif),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# ====================
# === Load Records ===
# ====================

def load_records(files):
    raw_dataset = tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)

    feature_desc = {
        "lead_msg": tf.io.FixedLenFeature([512], tf.int64),
        "lead_dif": tf.io.FixedLenFeature([512], tf.int64),
        "acmp_msg": tf.io.FixedLenFeature([512], tf.int64),
        "acmp_dif": tf.io.FixedLenFeature([512], tf.int64),
    }

    def _parse_function(example_proto):
        dictionary = tf.io.parse_single_example(example_proto, feature_desc)
        return tf.stack(
            [tf.cast(dictionary["lead_msg"], dtype=D_TYPE),
             tf.cast(dictionary["lead_dif"], dtype=D_TYPE),
             tf.cast(dictionary["acmp_msg"], dtype=D_TYPE),
             tf.cast(dictionary["acmp_dif"], dtype=D_TYPE),
             ])

    return raw_dataset.map(_parse_function)


# ====================
# === Tokenization ===
# ====================

class Tokenizer:
    """ Tokenizer for sequences.

    Ranges:
    [0        ] ... padding
    [1        ] ... start
    [2        ] ... stop
    [3   - 26 ] ... wait
    [27  - 114] ... note on
    [115 - 202] ... note off
    [203 - 217] ... time signature

    """

    def __init__(self, skip_time_signature=False) -> None:
        super().__init__()
        self.wait_buffer = 0
        self.flags = dict()
        self.flags["skip_time_signature"] = skip_time_signature

    def tokenize(self, entry):
        msg_type = entry["message_type"]

        if msg_type == MessageType.wait.value:
            value = int(entry["time"])
            self.wait_buffer += value
            return []
        elif msg_type == MessageType.note_on.value:
            shifter = 3 + 24
            value = int(entry["note"]) - 21
        elif msg_type == MessageType.note_off.value:
            shifter = 3 + 24 + 88
            value = int(entry["note"]) - 21
        elif msg_type == MessageType.time_signature.value:
            if self.flags.get("skip_time_signature", False):
                return []

            shifter = 3 + 24 + 88 + 88
            signature = (int(entry["numerator"]), int(entry["denominator"]))
            value = VALID_TIME_SIGNATURES.index(signature)
        else:
            raise UnexpectedValueException

        tokens = self.flush_wait_buffer()
        tokens.append(shifter + value)

        return tokens

    def tokenize_difficulty(self, difficulty):
        shifter = 3
        scaled_difficulty = math.floor(min(DIFFICULTY_VALUE_SCALE - 1, difficulty * DIFFICULTY_VALUE_SCALE))

        return shifter + scaled_difficulty

    def flush_wait_buffer(self):
        tokens = Tokenizer._flush_wait_buffer(self.wait_buffer)
        self.wait_buffer = 0
        return tokens

    @staticmethod
    def _flush_wait_buffer(wait_buffer):
        tokens = []

        while wait_buffer > 24:
            tokens.append(3 + (24 - 1))
            wait_buffer -= 24

        if wait_buffer > 0:
            tokens.append(3 + (wait_buffer - 1))

        return tokens


class Detokenizer:

    def __init__(self) -> None:
        super().__init__()
        self.wait_buffer = 0

    def detokenize(self, token):
        if token <= 2:
            return []
        elif 3 <= token <= 26:
            self.wait_buffer += token - 3 + 1
            return []
        elif 27 <= token <= 114:
            entry = {"message_type": "note_on", "note": token - 27 + 21}
        elif 115 <= token <= 202:
            entry = {"message_type": "note_off", "note": token - 115 + 21}
        elif 203 <= token <= 217:
            signature = VALID_TIME_SIGNATURES[token - 203]
            entry = {"message_type": "time_signature", "numerator": signature[0], "denominator": signature[1]}
        else:
            raise UnexpectedValueException

        entries = self.flush_wait_buffer()
        entries.append(entry)

        return entries

    def flush_wait_buffer(self):
        tokens = Detokenizer._flush_wait_buffer(self.wait_buffer)
        self.wait_buffer = 0
        return tokens

    @staticmethod
    def _flush_wait_buffer(wait_buffer):
        entries = []

        while wait_buffer > 24:
            entries.append({"message_type": "wait", "time": 24})
            wait_buffer -= 24

        if wait_buffer > 0:
            entries.append({"message_type": "wait", "time": wait_buffer})

        return entries
