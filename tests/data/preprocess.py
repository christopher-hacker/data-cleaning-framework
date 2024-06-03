"""Dummy preprocess script for testing."""

import pandas as pd


def preprocess() -> pd.DataFrame:
    """A dummy preprocessing function."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
