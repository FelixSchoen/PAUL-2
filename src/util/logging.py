import logging
import logging.config

from src.settings import ROOT_LOGGER
from src.util.util import get_src_root

initial_call = True


def get_logger(name):
    # Check if this is the first logger on this thread
    global initial_call

    if initial_call:
        root_path = get_src_root()
        logging.config.fileConfig(root_path + "/config/logging.conf")
        initial_call = False

    logger = logging.getLogger(ROOT_LOGGER + "." + name)

    return logger
