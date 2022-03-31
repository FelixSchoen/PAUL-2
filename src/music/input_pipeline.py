import copy
import gzip
import logging
import os.path
import pickle
import re
from multiprocessing import Pool

import mido
from sCoda import Composition, Bar

from src.util.util import chunks, flatten, file_exists, remove_random
from src.util.util_visualiser import get_message_lengths_and_difficulties


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
    bars = flatten(pool.map(_load_pickled_file, files))

    get_message_lengths_and_difficulties(bars)


def store_midi_files(directory: str) -> None:
    """ Loads the MIDI files from the drive, processes them, and stores the processed files.

    Args:
        directory: The directory to load the files from

    """
    files = []

    for dir_path, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            files.append((os.path.join(dir_path, filename), filename))

    files = remove_random(files, 0.9)
    files_chunks = chunks(files, 16)

    pool = Pool()
    pool.map(_load_and_store_midi_files, files_chunks)


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
        accompanying_bars = []

        # Copy the original bar in order to transpose it later on
        lead_unedited = [copy.copy(bar) for bar in base_bars[0]]
        accompanying_unedited = [copy.copy(bar) for bar in base_bars[1]]

        # Handle bars at the same time
        for lead_bar, accompanying_bar in zip(lead_unedited, accompanying_unedited):
            if lead_bar.transpose(transpose_by):
                lead_bar.set_difficulty()
            if accompanying_bar.transpose(transpose_by):
                accompanying_bar.set_difficulty()

            # Append transposed bars to the placeholder objects
            lead_bars.append(lead_bar)
            accompanying_bars.append(accompanying_bar)

        augmented_bars.append((lead_bars, accompanying_bars))

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


def _extract_bars_from_composition(composition: Composition) -> [([Bar], [Bar])]:
    """ Extracts pairs of up to 4 bars from the given composition.

    Args:
        composition: The composition to preprocess

    Returns: A tuple of two lists of bars, each of the same length

    """
    lead_track = composition.tracks[0]
    accompanying_track = composition.tracks[1]

    # Check that we start with the same number of bars
    assert len(lead_track.bars) == len(accompanying_track.bars)

    # Chunk the bars
    lead_chunked = list(chunks(lead_track.bars, 4))
    accompaniment_chunked = list(chunks(accompanying_track.bars, 4))

    # Zip the chunked bars
    zipped_chunks = list(zip(lead_chunked, accompaniment_chunked))

    return zipped_chunks


def _find_word(word, sentence) -> re.Match:
    """ Tries to find a word, not just a sequence of characters, in the given sentence

    Args:
        word: The word to look for
        sentence: The sentence in which the word may occur

    Returns: A `match` object

    """
    return re.compile(fr"\b({word})\b", flags=re.IGNORECASE).search(sentence)


def _load_and_store_midi_files(files) -> None:
    """ Loads and stores the MIDI files given.

    Args:
        files: A list of MIDI files to store

    """
    for filepath, filename in files:
        zip_file_path = f"D:/Documents/Coding/Repository/Badura/out/pickle/{filename[:-4]}.zip"

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
    accompanying_tracks = []
    meta_tracks = [0]

    # Open MIDI file
    midi_file = mido.MidiFile(file_path)

    # Parse track titles
    for t, track in enumerate(midi_file.tracks):
        if _find_word("right", track.name) is not None:
            lead_tracks.append(t)
        elif _find_word("left", track.name) is not None:
            accompanying_tracks.append(t)
        elif _find_word("pedal", track.name) is not None:
            meta_tracks.append(t)

    # Create composition from the found tracks
    composition = Composition.from_file(file_path, [lead_tracks, accompanying_tracks], meta_tracks)

    return composition


def _load_pickled_file(filepath_tuple) -> [([Bar], [Bar])]:
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
