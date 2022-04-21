import logging
import sys

loggers = dict()


def get_logger(name):
    if name in loggers:
        return loggers[name]

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    loggers[name] = logger

    return logger
