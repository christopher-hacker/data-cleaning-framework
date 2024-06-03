"""Dummy cleaners module for testing."""

from data_cleaning_framework import cleaner


@cleaner(columns=["my_field"])
def dummy_cleaner(value):
    """Dummy cleaner function."""
    return value + "_cleaned"
