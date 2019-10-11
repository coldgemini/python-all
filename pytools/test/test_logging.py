import logging
import sys
from logging.handlers import TimedRotatingFileHandler

LOG_FILE = "my_app.log"
fmt = "%(asctime)s — %(name)s — %(levelname)s | %(message)s"
datefmt = '%m-%d %H:%M:%S'
FORMATTER = logging.Formatter(fmt=fmt, datefmt=datefmt)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(log_file):
    file_handler = TimedRotatingFileHandler(log_file, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name, log_file=LOG_FILE):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(log_file))
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


def main():
    logger = get_logger("main", "tests.log")
    logger.setLevel(logging.DEBUG)
    logger.info("hahaha")


if __name__ == '__main__':
    main()
