import time

from src.util.logging import get_logger
from src.util.util import get_src_root, get_project_root


def test_logger():
    LOGGER = get_logger("root." + __name__)
    LOGGER.info("Test")
    time.sleep(2)


def test_root_path():
    print(get_project_root())
