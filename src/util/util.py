import gzip
import math
import pickle
import random
from pathlib import Path

from src.config.settings import DIFFICULTY_VALUE_SCALE


def chunk(lst, n):
    """ Yields chunks of size `n`

    Args:
        lst: The list to split into chunks
        n: Size of the chunks

    Returns: Consecutive chunks of the list

    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def convert_difficulty(difficulty):
    """ Converts the given difficulty to the internally used scale.

    Args:
        difficulty: The difficulty to convert

    Returns: A value conforming to the internal scale

    """
    if not 0 <= difficulty <= 1:
        print(difficulty)
    assert 0 <= difficulty <= 1

    return math.floor(min(DIFFICULTY_VALUE_SCALE - 1, difficulty * DIFFICULTY_VALUE_SCALE))


def file_exists(filepath):
    return Path(filepath).is_file()


def flatten(lst) -> []:
    """ Flattens the given list, reducing it by one level.

    Args:
        lst: The list to flatten

    Returns: The flattened list

    """
    return [item for sublist in lst for item in sublist]


def get_src_root() -> str:
    root_path = Path(__file__).parent.parent
    return str(root_path)


def get_prj_root() -> str:
    root_path = Path(__file__).parent.parent.parent
    return str(root_path)


def pickle_load(file_path):
    with gzip.open(file_path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, file_path):
    with gzip.open(file_path, "wb+") as f:
        pickle.dump(obj, f)


def remove_random(lst, n) -> []:
    """ Removes a percentage of the list's contents, randomly chosen.

    Args:
        lst: The list to remove items from
        n: The percentage to remove

    Returns: A new list, containing the sampled elements

    """
    return random.sample(lst, int(len(lst) * (1 - n)))
