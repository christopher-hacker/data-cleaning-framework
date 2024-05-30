"""Dummy cleaners module for testing."""

from data_cleaning_framework import cleaner


@cleaner(columns=["my_field"])
def dummy_cleaner(data):
    return data
