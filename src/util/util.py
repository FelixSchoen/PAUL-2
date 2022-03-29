def chunks(lst, n):
    """ Yields chunks of size `n`

    Args:
        lst: The list to split into chunks
        n: Size of the chunks

    Returns: Consecutive chunks of the list

    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flatten(lst):
    return [item for sublist in lst for item in sublist]
