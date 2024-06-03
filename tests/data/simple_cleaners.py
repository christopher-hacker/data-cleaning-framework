"""Dummy cleaners module for testing."""

from data_cleaning_framework import cleaner


@cleaner(columns=["col1"])
def dummy_cleaner(value):
    """Dummy cleaner function."""
    return value * 2
