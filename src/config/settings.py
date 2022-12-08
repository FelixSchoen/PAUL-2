from pathlib import Path

import tensorflow as tf

# Name of the root logger
from src.util.enumerations import NetworkType, NameSearchType

ROOT_LOGGER = "paul"


# =============
# === Paths ===
# =============

def get_project_root() -> str:
    root_path = Path(__file__).parent.parent.parent
    return str(root_path)


ROOT_PATH = get_project_root()

# Where to load raw MIDI files from
DATA_MIDI_INPUT_PATH = "D:/Documents/Coding/Repository/Paul/data/dataset"
DATA_MIDI_INPUT_PATH_SPARSE = "D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/sparse_data"

# Where to store processed bars for quicker loading
DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH = ROOT_PATH + "/out/bars/train"
DATA_BARS_VAL_OUTPUT_FOLDER_PATH = ROOT_PATH + "/out/bars/val"

# Where to store the datasets
DATA_TRAIN_OUTPUT_FILE_PATH = ROOT_PATH + "/out/dataset/data_train.tfrecords"
DATA_VAL_OUTPUT_FILE_PATH = ROOT_PATH + "/out/dataset/data_val.tfrecords"

PATH_OUT_PAUL = "/out/paul"
PATH_SAVED_MODEL = ROOT_PATH + PATH_OUT_PAUL + "/saved_model"
PATH_CHECKPOINT = ROOT_PATH + PATH_OUT_PAUL + "/checkpoint"
PATH_MIDI = ROOT_PATH + PATH_OUT_PAUL + "/gen"
PATH_TENSORBOARD = ROOT_PATH + "/out/tensorboard"

# =====================
# === Preprocessing ===
# =====================

# Length of tensors representing sequences
SEQUENCE_MAX_LENGTH = 512
# How many adjacent bars to consolidate into a tensor
CONSECUTIVE_BAR_MAX_LENGTH = 4
# Which time signatures to use
VALID_TIME_SIGNATURES = [(2, 2), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (9, 4), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8),
                         (8, 8), (9, 8), (12, 8), ]
# How many difficulty classes to use
DIFFICULTY_VALUE_SCALE = 10

# Vocabulary sizes
INPUT_VOCAB_SIZE_MLD = 218
LEAD_OUTPUT_VOCAB_SIZE = INPUT_VOCAB_SIZE_MLD
ACMP_OUTPUT_VOCAB_SIZE = LEAD_OUTPUT_VOCAB_SIZE - 15
INPUT_VOCAB_SIZE_DIF = DIFFICULTY_VALUE_SCALE + 3

# Token
D_TYPE = tf.uint8
PADDING_TOKEN = 0
START_TOKEN = 1
STOP_TOKEN = 2

# MIDI Track names
TRACK_NAME_SIGN = "Signature"
TRACK_NAME_META = "Meta"
TRACK_NAME_LEAD = "Piano Lead"
TRACK_NAME_ACMP = "Piano Acmp"
TRACK_NAME_UNKN = "Unknown"
TRACK_NAMES = [TRACK_NAME_SIGN, TRACK_NAME_META, TRACK_NAME_LEAD, TRACK_NAME_ACMP, TRACK_NAME_UNKN]
VALID_TRACK_NAMES = [(NameSearchType.phrase, "right", "left"),
                     (NameSearchType.word, "RH", "LH"),
                     (NameSearchType.phrase, "up", "down"),
                     (NameSearchType.word, "R", "L"),
                     (NameSearchType.phrase, "upper", "lower"),
                     (NameSearchType.phrase, "pianoUp", "pianoDown"),
                     (NameSearchType.phrase, "treble", "bass"),
                     (NameSearchType.word, "one", "two"),
                     (NameSearchType.word, "three", "four"),
                     (NameSearchType.phrase, "rechte", "linke"),
                     (NameSearchType.phrase, "rhStaff", "lhStaff"),
                     (NameSearchType.phrase, "new", "lower"),
                     (NameSearchType.word, "mel", "lower")]

# If a piece has more empty bars it is deleted from the dataset
MAX_PERCENTAGE_EMPTY_BARS = 0.4
# Whether unknown tracks should be used
ACCEPT_UNKNOWN_TRACKS = False

# ==================
# === Parameters ===
# ==================

TRAIN_VAL_SPLIT = 0.95
SHUFFLE_SEED = 6512924  # Felix
BUFFER_SIZE = 150000

EPOCHS = 10
MAX_CHECKPOINTS_TO_KEEP = 50
VAL_PER_BATCHES = 10/128

# Parameters explained in order of appearance:
# How often the encoder / decoder should be stacked
# Number of attention heads
# Dimension of the model
# Dimension of the feed-forward network
# Dropout rate applied after some operations
SETTINGS_LEAD_TRANSFORMER = {"BATCH_SIZE": 64, "NUM_LAYERS": 6, "NUM_HEADS": 4, "D_MODEL": 512, "DFF": 512,
                             "DROPOUT_RATE": 0.2, "OUTPUT_SIZE": LEAD_OUTPUT_VOCAB_SIZE, "TEMP": 0.6}
SETTINGS_ACMP_TRANSFORMER = {"BATCH_SIZE": 64, "NUM_LAYERS": 6, "NUM_HEADS": 4, "D_MODEL": 256, "DFF": 512,
                             "DROPOUT_RATE": 0.2, "OUTPUT_SIZE": ACMP_OUTPUT_VOCAB_SIZE, "TEMP": 0.4}

# ==================
# === Generation ===
# ==================

# Determines how many sequences are generated in parallel when generating
OUTPUT_DIMENSION = 16
# Maximum note length in ticks
MAXIMUM_NOTE_LENGTH = 24
# The amount of consecutive bars to generate
BARS_TO_GENERATE = 4
# The initial amount of consecutive bars to generate, before checking if they all conform to the same difficulty
BAR_GENERATION_STEP_SIZE = 2
