import tensorflow as tf
from pandas import DataFrame

from src.settings import SEQUENCE_MAX_LENGTH


def filter_length(src, trg):
    len1 = tf.shape(src)[1] if src is not None else 0
    len2 = tf.shape(trg)[2] if trg is not None else 0
    maximum = tf.maximum(len1, len2)
    return maximum < SEQUENCE_MAX_LENGTH


def dataframe_to_numeric_representation(data_frame: DataFrame):
    sequence = []

    # Pandas dataframe to list of tokens
    for k, row in data_frame.iterrows():
        token = Tokenizer.tokenize(row)
        if token is not None and token > 0:
            sequence.append(token)

    # Convert list to tensor
    tensor = tf.convert_to_tensor(sequence, dtype="int16")
    # Define how much to pad on each side
    paddings = [[0, SEQUENCE_MAX_LENGTH - tf.shape(tensor)[0]]]
    # Pad tensor
    tensor = tf.pad(tensor, paddings, "CONSTANT", constant_values=0)
    return tensor


class Tokenizer:

    @staticmethod
    def tokenize(entry):
        value = 0
        msg_type = entry["message_type"]

        if msg_type == "wait":
            shifter = 1
            value = int(entry["time"]) - 1
        elif msg_type == "note_on":
            shifter = 1 + 24
            value = int(entry["note"]) - 21
        elif msg_type == "note_off":
            shifter = 1 + 24 + 88
            value = int(entry["note"]) - 21
        else:
            shifter = -1

        return shifter + value

    @staticmethod
    def detokenize(token):
        if token <= 0:
            return None
        elif 1 <= token <= 24:
            return {"message_type": "wait", "time": token}
        elif 25 <= token <= 112:
            return {"message_type": "note_on", "note": token - 25 + 21}
        elif 113 <= token <= 200:
            return {"message_type": "note_off", "note": token - 113 + 21}
        else:
            assert False
