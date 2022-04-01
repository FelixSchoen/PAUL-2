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
from src.settings import SEQUENCE_MAX_LENGTH, DATA_COMPOSITIONS_PICKLE_OUTPUT_FILE_PATH, CONSECUTIVE_BAR_LENGTH, \
    BUFFER_SIZE, BATCH_SIZE
from src.util.util import chunks, flatten, file_exists


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


def store_midi_files(directory: str) -> None:
    """ Loads the MIDI files from the drive, processes them, and stores the processed files.

    Args:
        directory: The directory to load the files from

    """
    files = []

    # Handle all MIDI files in the given directory and subdirectories
    for dir_path, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            files.append((os.path.join(dir_path, filename), filename))

    # Separate into chunks in order to process in parallel
    files_chunks = chunks(files, 16)

    pool = Pool()
    pool.map(_store_midi_files, files_chunks)


def undefined(bars: [([Bar], [Bar])]):
    lead_sequences = np.array([], dtype=np.int16)
    lead_difficulties = np.array([], dtype=np.float16)
    acmp_sequences = np.array([], dtype=np.int16)
    acmp_difficulties = np.array([], dtype=np.float16)

    for entry in bars:
        lead_sequence, lead_difficulty = _bars_to_token(entry[0])
        acmp_sequence, acmp_difficulty = _bars_to_tensor(entry[1])

        # Filter long sequences
        if max(len(lead_sequence), len(lead_difficulty), len(acmp_sequence),
               len(acmp_difficulty)) <= SEQUENCE_MAX_LENGTH:
            lead_sequences = np.append(lead_sequences, _pad_array(lead_sequence))
            lead_difficulties = np.append(lead_difficulties, _pad_array(lead_difficulty))
            acmp_sequences = np.append(acmp_sequences, _pad_array(acmp_sequence))
            acmp_difficulties = np.append(acmp_difficulties, _pad_array(acmp_difficulty))

    print(lead_sequences)
    print(lead_difficulties)
    print(acmp_sequences)
    print(acmp_difficulties)

    # Construct dataset
    ds = tf.data.Dataset.from_tensor_slices((lead_sequences, lead_difficulties, acmp_sequences, acmp_difficulties))
    ds.cache()
    ds.shuffle(BUFFER_SIZE)
    ds.batch(BATCH_SIZE)
    ds.filter(filter_length)

    pass


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


def _bars_to_token(bars: [Bar]):
    sequence = np.array([])
    difficulties = np.array([])

    index = 0

    for bar in bars:
        data_frame = bar.to_relative_dataframe()

        # Sanity check
        assert len(data_frame) > 0

        # Pandas dataframe to list of tokens
        for k, row in data_frame.iterrows():
            try:
                token = Tokenizer.tokenize(row)
                sequence = np.append(sequence, token)
                difficulties = np.append(difficulties, bar.difficulty)
            except UnexpectedValueException:
                pass

    # Add start and stop messages
    sequence = np.insert(sequence, 0, -1)
    difficulties = np.insert(difficulties, 0, -1)
    sequence = np.append(sequence, -2)
    difficulties = np.append(difficulties, -2)

    return sequence, difficulties


def _bars_to_tensor(bars: [Bar]):
    sequence = []
    difficulties = []

    for bar in bars:
        data_frame = bar.to_relative_dataframe()

        # Sanity check
        assert len(data_frame) > 0

        # Pandas dataframe to list of tokens
        for k, row in data_frame.iterrows():
            try:
                token = Tokenizer.tokenize(row)
                sequence.append(token)
                difficulties.append(bar.difficulty)
            except UnexpectedValueException:
                pass

    # Add start and stop messages
    sequence.insert(0, -1)
    difficulties.insert(0, -1)
    sequence.append(-2)
    difficulties.append(-2)

    # Convert lists to tensors
    sequence_tensor = tf.convert_to_tensor(sequence, dtype="int16")
    difficulties_tensor = tf.convert_to_tensor(difficulties, dtype="float16")

    # Define paddings (how much to pad on each side)
    padding_sequence = [[0, SEQUENCE_MAX_LENGTH - tf.shape(sequence_tensor)[0]]]
    padding_difficulties = [[0, SEQUENCE_MAX_LENGTH - tf.shape(difficulties_tensor)[0]]]

    # Pad tensors
    sequence_tensor = tf.pad(sequence_tensor, padding_sequence, "CONSTANT", constant_values=0)
    difficulties_tensor = tf.pad(difficulties_tensor, padding_difficulties, "CONSTANT", constant_values=0)

    return sequence_tensor, difficulties_tensor


def _extract_bars_from_composition(composition: Composition) -> [([Bar], [Bar])]:
    """ Extracts pairs of up to `CONSECUTIVE_BAR_LENGTH` bars from the given composition.

    Args:
        composition: The composition to preprocess

    Returns: A tuple of two lists of bars, each of the same length

    """
    lead_track = composition.tracks[0]
    acmp_track = composition.tracks[1]

    # Check that we start with the same number of bars
    assert len(lead_track.bars) == len(acmp_track.bars)

    # Chunk the bars
    lead_chunked = list(chunks(lead_track.bars, CONSECUTIVE_BAR_LENGTH))
    acmp_chunked = list(chunks(acmp_track.bars, CONSECUTIVE_BAR_LENGTH))

    # Zip the chunked bars
    zipped_chunks = list(zip(lead_chunked, acmp_chunked))

    return zipped_chunks


def _find_word(word, sentence) -> re.Match:
    """ Tries to find a word, not just a sequence of characters, in the given sentence

    Args:
        word: The word to look for
        sentence: The sentence in which the word may occur

    Returns: A `match` object

    """
    return re.compile(fr"\b({word})\b", flags=re.IGNORECASE).search(sentence)


def _store_midi_files(files) -> None:
    """ Loads and stores the MIDI files given.

    Args:
        files: A list of MIDI files to store

    """
    for filepath, filename in files:
        zip_file_path = DATA_COMPOSITIONS_PICKLE_OUTPUT_FILE_PATH.format(filename[:-4])

        if file_exists(zip_file_path):
            logging.info(f"Skipping {filename}")
            continue

        logging.info(f"Processing {filename}...")

        # Loading the composition from the file
        composition = _load_composition(filepath)

        # Extracting bars that belong together
        extracted_bars = _extract_bars_from_composition(composition)

        # Add difficulty values
        _calculate_difficulty(extracted_bars)

        # Augment bars
        augmented_bars = flatten(list(map(_augment_bar, extracted_bars)))

        with gzip.open(zip_file_path, "wb+") as f:
            pickle.dump(augmented_bars, f)


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


def _pad_array(array, final_length: int = SEQUENCE_MAX_LENGTH) -> np.array:
    if len(array) >= final_length:
        return array
    return np.pad(array, (0, SEQUENCE_MAX_LENGTH - array.shape[0]), "constant")


# TODO Update
def filter_length(*args):
    length = 0

    for i in range(0, len(args)):
        print(tf.shape(args[i]))
        # length = max(length, tf.shape(args[i])[] if args[i] is not None else 0)

    # len1 = tf.shape(src)[1] if src is not None else 0
    # len2 = tf.shape(trg)[2] if trg is not None else 0
    # maximum = tf.maximum(len1, len2)
    return length <= SEQUENCE_MAX_LENGTH


class Tokenizer:
    """ Tokenizer for sequences.

    Ranges:
    [-2]        ... stop
    [-1]        ... start
    [0]         ... padding
    [1 - 24]    ... wait
    [25 - 112]  ... note on
    [113 - 200] ... note off

    """

    @staticmethod
    def tokenize(entry):
        msg_type = entry["message_type"]

        if msg_type == "wait":
            shifter = 1
            value = int(entry["time"]) - 1
        elif msg_type == "note_on":
            shifter = 1 + 24
            value = int(entry["note"]) - 21
        elif msg_type == "note_off":
            shifter = 1 + 24 + 88
            value = int(entry["note"]) - 21
        else:
            raise UnexpectedValueException

        return shifter + value

    @staticmethod
    def detokenize(token):
        if token <= 0:
            return None
        elif 1 <= token <= 24:
            return {"message_type": "wait", "time": token}
        elif 25 <= token <= 112:
            return {"message_type": "note_on", "note": token - 25 + 21}
        elif 113 <= token <= 200:
            return {"message_type": "note_off", "note": token - 113 + 21}
        else:
            raise UnexpectedValueException
