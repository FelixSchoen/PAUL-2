import copy
import math
import os.path
import random
import re
from multiprocessing import Pool

import mido
import numpy as np
import tensorflow as tf
from sCoda import Composition, Bar
from sCoda.elements.message import MessageType

from src.config.settings import SEQUENCE_MAX_LENGTH, CONSECUTIVE_BAR_MAX_LENGTH, \
    VALID_TIME_SIGNATURES, DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH, \
    DATA_TRAIN_OUTPUT_FILE_PATH, START_TOKEN, STOP_TOKEN, D_TYPE, DIFFICULTY_VALUE_SCALE, TRAIN_VAL_SPLIT, \
    DATA_BARS_VAL_OUTPUT_FOLDER_PATH
from src.exception.exceptions import UnexpectedValueException
from src.util.logging import get_logger
from src.util.util import chunks, flatten, file_exists, pickle_save, pickle_load


def load_and_store_records(input_dir=DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH, output_path=DATA_TRAIN_OUTPUT_FILE_PATH):
    bars = load_stored_bars(directory=input_dir)
    data_rows = map(_bar_tuple_to_token_tuple, bars)
    data_rows = filter(_filter_length, data_rows)
    data_rows = map(_pad_tuples, data_rows)

    pool = Pool()
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for example in pool.map(_serialize_example, list(data_rows)):
            writer.write(example)


def load_dataset_from_records(files):
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


def load_stored_bars(directory=DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH) -> [([Bar], [Bar])]:
    """ Loads the stored bars from the given directory into memory.

    Args:
        directory: Directory to a path full of zipped pickle files

    Returns: The loaded bars, as a list of tuples, each tuple containing an equal amount of bars for the right and
    left hand

    """
    files = []

    for dir_path, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".zip")]:
            files.append((os.path.join(dir_path, filename), filename))

    # Separate into chunks in order to process in parallel
    files_chunks = list(chunks(files, 16))

    pool = Pool()
    bars = flatten(pool.map(_load_stored_bars, files_chunks))

    return bars


def load_midi_files(directory: str, flags=None) -> list:
    """ Loads the MIDI files from the drive, processes them, and stores the processed files.

    Applies a train / validation split according to the percentage given in the settings, and stores the processed bars
    in different directories regarding their split.

    Args:
        directory: The directory to load the files from
        flags: A list of flags for the processing of files

    Returns: The loaded files

    """
    if flags is None:
        flags = []

    files = []

    # Handle all MIDI files in the given directory and subdirectories
    for dir_path, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            files.append((os.path.join(dir_path, filename), filename))

    # Separate into chunks in order to process in parallel
    files_chunks = list(chunks(files, 1))

    pool = Pool()
    loaded_files = pool.starmap(_load_midi_files,
                                zip(files_chunks, [copy.copy(flags) for _ in range(0, len(files_chunks))]))

    return flatten(loaded_files)


def _augment_bar(base_bars: ([Bar], [Bar])) -> [([Bar], [Bar])]:
    """ Augment the given bars in order to create a larger training set.

    Transposes the given bars by each value in the range of `[-5, 6]`.

    Args:
        base_bars: A tuple of bars to augment

    Returns: A list of all the augmentations

    """
    augmented_bars = []

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

        augmented_bars.append((lead_bars, acmp_bars))

    return augmented_bars


def _calculate_difficulty(bar_chunks: [([Bar], [Bar])]) -> [([Bar], [Bar])]:
    """ Adds difficulty values to all the given bars.

    Args:
        bar_chunks: The bars to calculate the difficulty for

    """
    for chunk in bar_chunks:
        for bar in chunk[0]:
            bar.difficulty()
        for bar in chunk[1]:
            bar.difficulty()

    return bar_chunks


def _bar_tuple_to_token_tuple(bars: ([Bar], [Bar])):
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


def _extract_bars_from_composition(composition: Composition) -> [([Bar], [Bar])]:
    """ Extracts pairs of up to `CONSECUTIVE_BAR_LENGTH` bars from the given composition.

    Checks for each bar whether its time signature is in the list of supported time signatures. If not, breaks up the
    bar at the given position, ignoring the one with the unsupported signature.

    Args:
        composition: The composition to preprocess

    Returns: A tuple of two lists of bars, each of the same length

    """
    logger = get_logger(__name__)
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
            logger.debug("Unknown signature, breaking up bars.")
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


def _filter_length(to_filter):
    length = max([tf.shape(x)[0] for x in to_filter])
    return length <= SEQUENCE_MAX_LENGTH - 2  # Leave space for start and stop messages


def _find_word(word, sentence) -> re.Match:
    """ Tries to find a word, not just a sequence of characters, in the given sentence

    Args:
        word: The word to look for
        sentence: The sentence in which the word may occur

    Returns: A `match` object

    """
    return re.compile(fr"\b({word})\b", flags=re.IGNORECASE).search(sentence)


def _load_midi_files(files, flags: list) -> list:
    """ Loads and stores the MIDI files given.

    Args:
        files: A list of MIDI files to store
        flags: A list of flags for the operations

    Returns: A list of processed compositions

    """
    logger = get_logger(__name__)
    processed_files = []

    for filepath, filename in files:
        train_zip_file_path = (DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH + "/{0}.zip").format(filename[:-4])
        val_zip_file_path = (DATA_BARS_VAL_OUTPUT_FOLDER_PATH + "/{0}.zip").format(filename[:-4])

        if file_exists(train_zip_file_path) and "skip_skip" not in flags:
            logger.info(f"Skipping {filename}.")
            continue

        logger.info(f"Processing {filename}...")

        # Loading the composition from the file
        composition = _load_composition(filepath)

        # Extracting bars that belong together
        extracted_bars = _extract_bars_from_composition(composition)

        # Add difficulty values
        if "skip_difficulty" not in flags:
            _calculate_difficulty(extracted_bars)

        # Shuffle bars
        random.shuffle(extracted_bars)

        split_point = int((len(extracted_bars) + 1) * TRAIN_VAL_SPLIT)
        train_extracted_bars = extracted_bars[:split_point]
        val_extracted_bars = extracted_bars[split_point:]

        # Augment bars
        train_augmented_bars = flatten(list(map(_augment_bar, train_extracted_bars)))
        val_augmented_bars = flatten(list(map(_augment_bar, val_extracted_bars)))

        # Store to file
        if "skip_store" not in flags:
            pickle_save(train_augmented_bars, train_zip_file_path)
            pickle_save(val_augmented_bars, val_zip_file_path)

        processed_files.append(train_augmented_bars)
        processed_files.append(val_augmented_bars)

        logger.info(f"Finished processing {filename}.")

    return processed_files


def _load_composition(file_path: str) -> Composition:
    """ Loads a compositions from a given MIDI file

    Args:
        file_path: Path to the MIDI file

    Returns: A `Composition` object

    """
    lead_tracks = []
    acmp_tracks = []
    meta_tracks = [0]

    # Open MIDI file
    midi_file = mido.MidiFile(file_path)

    # Parse track titles
    for t, track in enumerate(midi_file.tracks):
        if _find_word("right", track.name) is not None:
            lead_tracks.append(t)
        elif _find_word("left", track.name) is not None:
            acmp_tracks.append(t)
        elif _find_word("pedal", track.name) is not None:
            meta_tracks.append(t)

    # Create composition from the found tracks
    composition = Composition.from_file(file_path, [lead_tracks, acmp_tracks], meta_tracks)

    return composition


def _load_stored_bars(filepaths_tuple) -> [([Bar], [Bar])]:
    """ Loads a pickled and preprocessed composition.

    Args:
        filepaths_tuple: A list of tuples consisting of the filepath and name of the file

    Returns: The loaded preprocessed composition

    """
    logger = get_logger(__name__)
    loaded_files = []

    for filepath, filename in filepaths_tuple:
        logger.info(f"Loading {filename}...")

        from_pickle = pickle_load(filepath)

        loaded_files.append(from_pickle)

    return flatten(loaded_files)


def _pad_tuples(tuples_to_pad):
    results = []

    for ele in tuples_to_pad:
        results.append(np.pad(ele, (0, SEQUENCE_MAX_LENGTH - ele.shape[0]), "constant"))

    return results


def _serialize_example(entry):
    lead_msg, lead_dif, acmp_msg, acmp_dif = entry
    record = _tensor_to_record(lead_msg, lead_dif, acmp_msg, acmp_dif)
    return record.SerializeToString()


def _tensor_to_record(lead_msg, lead_dif, acmp_msg, acmp_dif):
    def _int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    feature = {
        "lead_msg": _int_feature(lead_msg),
        "lead_dif": _int_feature(lead_dif),
        "acmp_msg": _int_feature(acmp_msg),
        "acmp_dif": _int_feature(acmp_dif),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


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
