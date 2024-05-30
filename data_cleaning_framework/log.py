"""Contains logging utilities for the data cleaning framework."""

import logging
from typing import Optional


def get_logger(scenario: Optional[str] = None) -> logging.Logger:
    """Creates a logger for the clean_data script."""
    # Create a logger
    logger_obj = logging.getLogger(__name__)
    logger_obj.setLevel(logging.DEBUG)

    # Create a file handler for outputting to "clean_data.log"
    if scenario is not None:
        file_handler = logging.FileHandler(f"clean_data_{scenario}.log", mode="w")
    else:
        file_handler = logging.FileHandler("clean_data.log", mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger_obj.addHandler(file_handler)

    return logger_obj
