# Name of the root logger
import tensorflow as tf

from src.util.util import get_project_root

ROOT_LOGGER = "badura"

ROOT_PATH = get_project_root()

# Where to load raw MIDI files from
DATA_MIDI_INPUT_PATH = "D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/data"
# Where to store processed compositions for quicker loading
DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH = ROOT_PATH + "/out/pickle/compositions"
DATA_COMPOSITIONS_PICKLE_OUTPUT_FILE_PATH = DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH + "/{0}.zip"
DATA_SET_OUTPUT_FILE_PATH = ROOT_PATH + "/out/dataset/data.tfrecords"

PATH_SAVED_MODEL = ROOT_PATH + "/out/badura/saved_model"
PATH_CHECKPOINT = ROOT_PATH + "/out/badura/checkpoint"
PATH_TENSORBOARD = ROOT_PATH + "/out/tensorboard"

# Length of tensors representing sequences
SEQUENCE_MAX_LENGTH = 512
# How many adjacent bars to consolidate into a tensor
CONSECUTIVE_BAR_MAX_LENGTH = 4
# Which time signatures to use
VALID_TIME_SIGNATURES = [(2, 2), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (9, 4), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8),
                         (8, 8), (9, 8), (12, 8), ]
DIFFICULTY_VALUE_SCALE = 10

# Vocabulary sizes
INPUT_VOCAB_SIZE_MLD = 218
LEAD_OUTPUT_VOCAB_SIZE = INPUT_VOCAB_SIZE_MLD
ACMP_OUTPUT_VOCAB_SIZE = LEAD_OUTPUT_VOCAB_SIZE - 15
INPUT_VOCAB_SIZE_DIF = DIFFICULTY_VALUE_SCALE + 3

# Token
D_TYPE = tf.uint8
START_TOKEN = 1
STOP_TOKEN = 2

# ==================
# === Parameters ===
# ==================

TRAIN_VAL_SPLIT = 0.95
BATCH_SIZE = 1
SHUFFLE_SEED = 6512924  # Felix
BUFFER_SIZE = 150000

EPOCHS = 1  # 32

# How often the encoder / decoder should be stacked
NUM_LAYERS = 6  # 4
# Number of attention heads
NUM_HEADS = 8
# Dimension of the model
D_MODEL = 256  # 128
# Dimension of the feed-forward network
DFF = 1024  # 512
# Dropout rate applied after some operations
DROPOUT_RATE = 0.1