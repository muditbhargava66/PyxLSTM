"""
Configuration Utility
Author: Mudit Bhargava

This module provides utilities for loading and managing configuration settings.
"""

import yaml

class Config:
    def __init__(self, config_file):
        """
        Initialize the Config object.

        Args:
            config_file (str): Path to the YAML configuration file
        """
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

    def __getattr__(self, key):
        """
        Get a configuration value.

        Args:
            key (str): The configuration key

        Returns:
            The value associated with the key
        """
        return self.config.get(key)

def load_config(config_file):
    """
    Load a configuration file.

    Args:
        config_file (str): Path to the YAML configuration file

    Returns:
        Config: A Config object containing the loaded configuration
    """
    return Config(config_file)