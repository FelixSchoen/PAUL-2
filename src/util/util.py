import random
from pathlib import Path


def chunks(lst, n):
    """ Yields chunks of size `n`

    Args:
        lst: The list to split into chunks
        n: Size of the chunks

    Returns: Consecutive chunks of the list

    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def file_exists(filepath):
    return Path(filepath).is_file()


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def remove_random(lst, n):
    return random.sample(lst, int(len(lst) * (1 - n)))
