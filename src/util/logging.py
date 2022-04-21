import logging
import logging.config

from src.settings import ROOT_LOGGER

initial_call = True


def get_logger(name):
    # Check if this is the first logger on this thread
    global initial_call

    if initial_call:
        logging.config.fileConfig("config/logging.conf")
        initial_call = False

    logger = logging.getLogger(ROOT_LOGGER + "." + name)

    return logger
