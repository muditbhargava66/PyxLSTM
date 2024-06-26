"""
Logging Utility
Author: Mudit Bhargava

This module provides utilities for setting up and managing logging.
"""

import logging

def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        log_file (str, optional): Path to the log file. If None, log to console only.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG)
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger

    Returns:
        logging.Logger: A logger object
    """
    return logging.getLogger(name)