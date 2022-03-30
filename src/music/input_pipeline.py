import copy
import logging
import os.path
import re
from multiprocessing import Pool

import mido
from sCoda import Composition, Bar

from src.util.util import chunks, flatten, remove_random
from src.util.util_visualiser import get_message_lengths_and_difficulties


def load_midi_files(directory: str):
    files = []

    for dir_path, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            files.append(os.path.join(dir_path, filename))

    # files = remove_random(files, 0.9)

    for file in files:
        logging.info(f"Loading {file}...")
        composition = _load_composition(file)
        preprocessed_bars = _preprocess_composition(composition)
        _calculate_difficulty(preprocessed_bars)
        list(map(_augment_bar, preprocessed_bars))

    # pool = Pool()
    #
    # bars = flatten(pool.map(_load, list(chunks(files, int(len(files) / 16)))))
    # get_message_lengths_and_difficulties(bars)


def _load(files: [str]):
    # Load compositions
    compositions = list(map(_load_composition, files))
    # Preprocess bars
    preprocessed_bars = flatten(list(map(_preprocess_composition, compositions)))
    # Set difficulty of bars
    preprocessed_bars = flatten(list(map(_calculate_difficulty, chunks(preprocessed_bars, 32))))

    # Augment bars after difficulty calculation
    augmented_bars = flatten(list(map(_augment_bar, preprocessed_bars)))

    return augmented_bars


def _load_composition(file_path: str):
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


def _preprocess_composition(composition: Composition) -> [([Bar], [Bar])]:
    """ Preprocesses the given composition, extracting pairs of up to 4 bars

    Sets the `difficulty` value for the bars as well, efficiently calculating it for all the bars since this method
    is parallelised.

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


def _augment_bar(bars: ([Bar], [Bar])) -> [([Bar], [Bar])]:
    """ Augment the given bars in order to create a larger training set

    Transposes the given bars by each value in the range of `[-5, 6]`.

    Args:
        bars: A tuple of bars to augment

    Returns: A list of all the augmentations

    """
    augmented_bars = []

    for transpose_by in range(-5, 7):
        lead_bars = []
        accompanying_bars = []

        # Copy the original bar in order to transpose it later on
        lead_unedited = [copy.copy(bar) for bar in bars[0]]
        accompanying_unedited = [copy.copy(bar) for bar in bars[1]]

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


def _calculate_difficulty(bar_chunks: [([Bar], [Bar])]) -> None:
    for chunk in bar_chunks:
        for bar in chunk[0]:
            bar.set_difficulty()
        for bar in chunk[1]:
            bar.set_difficulty()

    return bar_chunks


def _find_word(word, sentence):
    """ Tries to find a word, not just a sequence of characters, in the given sentence

    Args:
        word: The word to look for
        sentence: The sentence in which the word may occur

    Returns: A `match` object

    """
    return re.compile(fr"\b({word})\b", flags=re.IGNORECASE).search(sentence)
