import copy
import gzip
import logging
import os.path
import pickle
import re
from multiprocessing import Pool

import mido
import numpy as np
import tensorflow as tf
from sCoda import Composition, Bar

from src.exception.exceptions import UnexpectedValueException
from src.settings import SEQUENCE_MAX_LENGTH, DATA_COMPOSITIONS_PICKLE_OUTPUT_FILE_PATH, CONSECUTIVE_BAR_MAX_LENGTH, \
    BUFFER_SIZE, BATCH_SIZE, VALID_TIME_SIGNATURES, DIFFICULTY_VALUE_SCALE
from src.util.util import chunks, flatten, file_exists


def load_dataset(bars: [([Bar], [Bar])]):
    data_rows = map(_bar_tuple_to_token_tuple, bars)
    data_rows = filter(_filter_length, data_rows)
    data_rows = map(_pad_tuples, data_rows)

    # Construct dataset
    ds = tf.data.Dataset.from_tensor_slices(list(data_rows)) \
        .cache() \
        .shuffle(BUFFER_SIZE) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)

    return ds


def load_stored_bars(directory: str) -> [([Bar], [Bar])]:
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

    pool = Pool()
    bars = flatten(pool.map(_load_stored_bars, files))

    return bars


def load_midi_files(directory: str, flags=None) -> list:
    """ Loads the MIDI files from the drive, processes them, and stores the processed files.

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
    files_chunks = list(chunks(files, 16))

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
            if lead_bar.transpose(transpose_by):
                lead_bar.set_difficulty()
            if acmp_bar.transpose(transpose_by):
                acmp_bar.set_difficulty()

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
            bar.set_difficulty()
        for bar in chunk[1]:
            bar.set_difficulty()

    return bar_chunks


def _bar_tuple_to_token_tuple(bars: ([Bar], [Bar])):
    lead_seq, lead_dif, acmp_seq, acmp_dif = [], [], [], []

    for i, (seq, dif) in enumerate([(lead_seq, lead_dif), (acmp_seq, acmp_dif)]):
        for bar in bars[i]:
            data_frame = bar.to_relative_dataframe()

            # Sanity check
            assert len(data_frame) > 0

            tokenizer = Tokenizer()

            # Pandas dataframe to list of tokens
            for _, row in data_frame.iterrows():
                try:
                    tokens = tokenizer.tokenize(row)
                    seq.extend(tokens)
                    dif.extend([int(bar.difficulty * DIFFICULTY_VALUE_SCALE + 1) for _ in range(0, len(tokens))])
                except UnexpectedValueException:
                    pass

            # Append trailing wait messages
            tokens = tokenizer.flush_wait_buffer()
            seq.extend(tokens)
            dif.extend([int(bar.difficulty * DIFFICULTY_VALUE_SCALE + 1) for _ in range(0, len(tokens))])

        # Add start and stop messages
        seq.insert(0, -1)
        dif.insert(0, -1)
        seq.append(-2)
        dif.append(-2)

    return tf.convert_to_tensor(lead_seq, dtype=tf.int16), tf.convert_to_tensor(lead_dif, dtype=tf.int16), \
           tf.convert_to_tensor(acmp_seq, dtype=tf.int16), tf.convert_to_tensor(acmp_dif, dtype=tf.int16)


def _extract_bars_from_composition(composition: Composition) -> [([Bar], [Bar])]:
    """ Extracts pairs of up to `CONSECUTIVE_BAR_LENGTH` bars from the given composition.

    Checks for each bar whether its time signature is in the list of supported time signatures. If not, breaks up the
    bar at the given position, ignoring the one with the unsupported signature.

    Args:
        composition: The composition to preprocess

    Returns: A tuple of two lists of bars, each of the same length

    """
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
            logging.info("Unknown signature, breaking up bars")
            remaining = CONSECUTIVE_BAR_MAX_LENGTH

            if len(lead_current) > 0:
                lead_chunked.append(lead_current)
                acmp_chunked.append(acmp_current)

            lead_current = []
            acmp_current = []

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

    lead_chunked = list(chunks(lead_track.bars, CONSECUTIVE_BAR_MAX_LENGTH))
    acmp_chunked = list(chunks(acmp_track.bars, CONSECUTIVE_BAR_MAX_LENGTH))

    # Zip the chunked bars
    zipped_chunks = list(zip(lead_chunked, acmp_chunked))

    return zipped_chunks


def _filter_length(to_filter):
    length = max([tf.shape(x)[0] for x in to_filter])
    return length <= SEQUENCE_MAX_LENGTH


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
    processed_files = []

    for filepath, filename in files:
        zip_file_path = DATA_COMPOSITIONS_PICKLE_OUTPUT_FILE_PATH.format(filename[:-4])

        if file_exists(zip_file_path) and "skip_skip" not in flags:
            logging.info(f"Skipping {filename}")
            continue

        logging.info(f"Processing {filename}...")

        # Loading the composition from the file
        composition = _load_composition(filepath)

        # Extracting bars that belong together
        extracted_bars = _extract_bars_from_composition(composition)

        # Add difficulty values
        if "skip_difficulty" not in flags:
            _calculate_difficulty(extracted_bars)

        # Augment bars
        augmented_bars = flatten(list(map(_augment_bar, extracted_bars)))

        # Store to file
        if "skip_store" not in flags:
            with gzip.open(zip_file_path, "wb+") as f:
                pickle.dump(augmented_bars, f)

        processed_files.append(augmented_bars)

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


def _load_stored_bars(filepath_tuple) -> [([Bar], [Bar])]:
    """ Loads a pickled and preprocessed composition.

    Args:
        filepath_tuple: A tuple consisting of the filepath and name of the file

    Returns: The loaded preprocessed composition

    """
    filepath, filename = filepath_tuple

    print(f"Loading {filename}...")
    with gzip.open(filepath, "rb") as f:
        from_pickle = pickle.load(f)

    return from_pickle


def _pad_tuples(tuples_to_pad):
    results = []

    for ele in tuples_to_pad:
        results.append(np.pad(ele, (0, SEQUENCE_MAX_LENGTH - ele.shape[0]), "constant"))

    return results


class Tokenizer:
    """ Tokenizer for sequences.

    Ranges:
    [-2       ] ... stop
    [-1       ] ... start
    [0        ] ... padding
    [1   - 24 ] ... wait
    [25  - 112] ... note on
    [113 - 200] ... note off
    [201 - 215] ... time signature

    """

    def __init__(self) -> None:
        super().__init__()
        self.wait_buffer = 0

    def tokenize(self, entry):
        msg_type = entry["message_type"]

        if msg_type == "wait":
            shifter = 1
            value = int(entry["time"]) - 1
            self.wait_buffer += shifter + value
            return []
        elif msg_type == "note_on":
            shifter = 1 + 24
            value = int(entry["note"]) - 21
        elif msg_type == "note_off":
            shifter = 1 + 24 + 88
            value = int(entry["note"]) - 21
        elif msg_type == "time_signature":
            shifter = 1 + 24 + 88 + 88
            signature = (int(entry["numerator"]), int(entry["denominator"]))
            value = VALID_TIME_SIGNATURES.index(signature)
        else:
            raise UnexpectedValueException

        tokens = self.flush_wait_buffer()
        tokens.append(shifter + value)

        return tokens

    def flush_wait_buffer(self):
        tokens = Tokenizer._flush_wait_buffer(self.wait_buffer)
        self.wait_buffer = 0
        return tokens

    @staticmethod
    def _flush_wait_buffer(wait_buffer):
        tokens = []

        while wait_buffer > 24:
            tokens.append(1 + (24 - 1))
            wait_buffer -= 24

        if wait_buffer > 0:
            tokens.append(1 + (wait_buffer - 1))

        return tokens


class Detokenizer:

    def __init__(self) -> None:
        super().__init__()
        self.wait_buffer = 0

    def detokenize(self, token):
        if token <= 0:
            return []
        elif 1 <= token <= 24:
            self.wait_buffer += token - 1 + 1
            return []
        elif 25 <= token <= 112:
            entry = {"message_type": "note_on", "note": token - 25 + 21}
        elif 113 <= token <= 200:
            entry = {"message_type": "note_off", "note": token - 113 + 21}
        elif 201 <= token <= 215:
            signature = VALID_TIME_SIGNATURES[token - 201]
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
