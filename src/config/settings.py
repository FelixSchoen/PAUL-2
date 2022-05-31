from pathlib import Path

import tensorflow as tf

# Name of the root logger
from src.util.enumerations import NetworkType

ROOT_LOGGER = "paul"


# =============
# === Paths ===
# =============

def get_project_root() -> str:
    root_path = Path(__file__).parent.parent.parent
    return str(root_path)


ROOT_PATH = get_project_root()

# Where to load raw MIDI files from
DATA_MIDI_INPUT_PATH = "D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/data"
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

# ==================
# === Parameters ===
# ==================

TRAIN_VAL_SPLIT = 0.9
SHUFFLE_SEED = 6512924  # Felix
BUFFER_SIZE = 150000

EPOCHS = 6
OUTPUT_DIMENSION = 16

# Parameters explained in order of appearance:
# How often the encoder / decoder should be stacked
# Number of attention heads
# Dimension of the model
# Dimension of the feed-forward network
# Dropout rate applied after some operations
SETTINGS_LEAD_TRANSFORMER = {"BATCH_SIZE": 128, "NUM_LAYERS": 6, "NUM_HEADS": 4, "D_MODEL": 256, "DFF": 256,
                             "DROPOUT_RATE": 0.2, "OUTPUT_SIZE": LEAD_OUTPUT_VOCAB_SIZE}
SETTINGS_ACMP_TRANSFORMER = {"BATCH_SIZE": 128, "NUM_LAYERS": 6, "NUM_HEADS": 4, "D_MODEL": 64, "DFF": 128,
                             "DROPOUT_RATE": 0.2, "OUTPUT_SIZE": ACMP_OUTPUT_VOCAB_SIZE}


def set_settings(settings, network):
    if network == NetworkType.lead:
        global SETTINGS_LEAD_TRANSFORMER
        SETTINGS_LEAD_TRANSFORMER = settings
    elif network == NetworkType.acmp:
        global SETTINGS_ACMP_TRANSFORMER
        SETTINGS_ACMP_TRANSFORMER = settings
