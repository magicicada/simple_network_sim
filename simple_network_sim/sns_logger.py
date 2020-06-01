# -*- coding: utf-8 -*-
"""Provides logging support for the simple_network_sim module.

This approach define a package-level logger that works as follows:

1. Import sns_logger from any package module - this will instantiate
a new logging.Logger instance as sns_logger.logger and use the
setup_logger() function to set the default formatting. From the point
of first import, the sns_logger.logger instance will be available
for logging.

2. To modify logger configuration (e.g. on the basis of CLI options),
call setup_logger() with an argparse.Namespace that includes the
fields 'logfile', 'verbose', and 'debug'. This will change global
settings for this logger. For example, setting a logfile in the entry
point for a script will make that logfile available to all logger
calls (where the sns_logger.logger is used). Calling setup_logger()
should - for consistency - be done before any information is written
to the logger.

To use the package-level logger, you might use

from . import sns_logger
logger = sns_logger.logger

or

from simple_network_sim import sns_logger
logger = sns_logger.logger

NOTE: By default, the logger writes to sys.stderr and only reports
logger.WARNING and above.
"""

import logging
import logging.config
import sys

from argparse import Namespace
from pathlib import Path
from typing import Optional

def setup_logger(args: Optional[Namespace] = None) -> None:
    """Create and return a logging.Logger instance.
    
    :param args: argparse.Namespace
        args.logfile (pathlib.Path) is used to create a logfile if present
        args.verbose and args.debug control logging level to sys.stderr
    """
    # Dictionary to define logging configuration
    logconf = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                },
            },
        "handlers": {
            "stderr": {
                "level": "WARNING",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stderr"
                },
            },
        "loggers": {
            __package__: {
                "handlers": ["stderr"],
                "level": "DEBUG",
                },
            },
        }

    # If args.logpath is specified, add logfile
    if args is not None and args.logfile is not None:
        logdir = args.logfile.parents[0]
        # If the logfile is going in another directory, we must
        # create it/check if it's there
        try:
            if not logdir == Path.cwd():
                logdir.mkdir(exist_ok=True)
        except OSError:
            logger.error("Could not create %s for logging", logdir, exc_info=True)
            raise SystemExit(1)  # Substitute meaningful error code when known
        # Add logfile configuration
        logconf["handlers"]["logfile"] = {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": str(args.logfile),
            "encoding": "utf8",
            }
        logconf["loggers"][__package__]["handlers"].append("logfile")

    # Set STDERR/logfile levels if args.verbose/args.debug specified
    if args is not None and args.verbose:
        logconf["handlers"]["stderr"]["level"] = "INFO"
    elif args is not None and args.debug:
        logconf["handlers"]["stderr"]["level"] = "DEBUG"
        logconf["handlers"]["logfile"]["level"] = "DEBUG"

    # Configure logger
    logging.config.dictConfig(logconf)

# Provide a package-level logging.Logger object on import
logger = logging.getLogger(__package__)
setup_logger()