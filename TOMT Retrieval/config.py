import logging
import os
from logging import handlers


def supress_log(lib_name):
    logging.getLogger(lib_name).setLevel(logging.WARNING)


def add_common_args(parser):
    parser.add_argument(
        "--verbose", help="If set, show verbose logs", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)


def configure_logging(module, verbose, log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        logging.handlers.RotatingFileHandler(
            f"{log_dir}/{module}.log", maxBytes=1048576 * 5, backupCount=7),
        logging.StreamHandler()
    ]
    log_format = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    if verbose:
        logging.basicConfig(level=logging.DEBUG,
                            handlers=handlers, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=handlers, format=log_format)
