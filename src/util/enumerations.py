from enum import Enum


class NetworkType(Enum):
    lead = "lead"
    acmp = "acmp"


class NameSearchType(Enum):
    phrase = "phrase"
    word = "word"
