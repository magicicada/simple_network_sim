# -*- coding: utf-8 -*-
"""Provides logging for the simple_network_sim module."""

import logging

from argparse import Namespace
from typing import Optional

def build_logger(name: str = __name__, args: Optional[Namespace] = None) -> logging.Logger:
    """Create and return a logging.Logger instance.
    
    :param name: Name for the logger. Default: __name__
    :param args: argparse.Namespace; args.logfile (pathlib.Path) is used to create a logfile if present
    """
    # Instantiate logger
    logger = logging.getLogger(__name__)

    # Common format for logstrings
    log_formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

    # StreamHandler for default error handling (sys.stderr)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(log_formatter)
    err_handler.setLevel(logging.WARNING)
    logger.addHandler(err_handler)

    # Streamhandler for optional logfile
    if args is not None and args.logfile is not None:
        logdir - args.logfile.parents[0]
        try:
            # If the logfile is going in another directory, we must
            # create it/check if it's there
            if not logdir == Path.cwd():
                logdir.mkdir(exist_ok=True)
            logstream = args.logfile.open("w")
        except OSError:
            logger.error("Could not open %s for logging", args.logfile, exc_info=True)
            raise SystemExit(1)  # Substitute meaningful error code when known
    
        file_handler = logging.StreamHandler(logstream)
        file_handler.setFormatter(err_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


# If imported as a module, create a logging.Logger instance
logger = build_logger()