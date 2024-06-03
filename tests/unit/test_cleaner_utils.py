"""Tests for the cleaner_utils module."""

import pytest
from data_cleaning_framework.cleaner_utils import cleaner


def test_cleaner_no_methods_specified():
    """Test ValueError when no methods are specified."""
    with pytest.raises(
        ValueError,
        match="At least one of `columns`, `dtypes`, or `dataframe_wise` must be specified.",
    ):

        @cleaner()
        def dummy_cleaner(df):
            return df


def test_cleaner_multiple_methods_specified():
    """Test ValueError when multiple methods are specified."""
    with pytest.raises(
        ValueError,
        match="Only one of `columns`, `dtypes`, or `dataframe_wise` can be specified.",
    ):

        @cleaner(columns=["col1"], dtypes=[int])
        def dummy_cleaner(df):
            return df
