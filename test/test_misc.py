import time
from logging import getLogger

from src.util.logging import get_logger


def test_logger():
    LOGGER = get_logger("root." + __name__)
    LOGGER.info("Test")
    time.sleep(2)
