import os.path
import re
from multiprocessing import Pool

import mido
from sCoda import Composition


def load_midi_files(directory: str):
    files = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mid")]:
            files.append(os.path.join(dirpath, filename))

    pool = Pool()
    compositions = pool.map(_load_composition, files)
    pool.map(_preprocess_compositions, compositions)

    print(compositions)


def _load_composition(file_path: str):
    lead_tracks = []
    accompanying_tracks = []
    meta_tracks = [0]

    midi_file = mido.MidiFile(file_path)

    for t, track in enumerate(midi_file.tracks):
        if _find_word("right", track.name) is not None:
            lead_tracks.append(t)
        elif _find_word("left", track.name) is not None:
            accompanying_tracks.append(t)
        elif _find_word("pedal", track.name) is not None:
            meta_tracks.append(t)

    composition = Composition.from_file(file_path, [lead_tracks, accompanying_tracks], meta_tracks)

    return composition


def _preprocess_compositions(composition: Composition):
    lead_track = composition.tracks[0]
    accompanying_track = composition.tracks[1]

    assert len(lead_track.bars) == len(accompanying_track.bars)

    zipped_bars = zip(lead_track.bars, accompanying_track.bars)


def _find_word(word, sentence):
    return re.compile(fr"\b({word})\b", flags=re.IGNORECASE).search(sentence)
