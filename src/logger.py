import logging
import os
from datetime import datetime

def my_logger(path):

    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    # logs_path = os.path.join(path, LOG_FILE)
    # os.makedirs(path, exist_ok=True)  # exist_ok means append in file

    LOG_FILE_PATH = os.path.join(path, LOG_FILE)

    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s",
        level=logging.INFO,
    )

    return logging


if __name__ == '__main__':
    logging.info("Logging has started")
