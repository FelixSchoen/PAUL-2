# Where to load raw MIDI files from
DATA_MIDI_INPUT_PATH = "D:/Drive/Documents/University/Master/4. Semester/Diplomarbeit/Resource/data"
# Where to store processed compositions for quicker loading
DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH = "D:/Documents/Coding/Repository/Badura/out/pickle/compositions"
DATA_COMPOSITIONS_PICKLE_OUTPUT_FILE_PATH = DATA_COMPOSITIONS_PICKLE_OUTPUT_FOLDER_PATH + "/{0}.zip"

# Length of tensors representing sequences
SEQUENCE_MAX_LENGTH = 512
# How many adjacent bars to consolidate into a tensor
CONSECUTIVE_BAR_LENGTH = 4

# ==================
# === Parameters ===
# ==================

BATCH_SIZE = 64
BUFFER_SIZE = 20000
# How often the encoder / decoder should be stacked
NUM_LAYERS = 4
# Number of attention heads
NUM_HEADS = 8
# Dimension of the model
D_MODEL = 128
# Dimension of the feed-forward network
DFF = 512
# Dropout rate applied after some operations
DROPOUT_RATE = 0.1