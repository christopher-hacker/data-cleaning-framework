"""Unit tests for the logging functionality in the clean_data module."""

from unittest.mock import patch
import pytest
import pandas as pd
from data_cleaning_framework.clean_data import log_processor


@log_processor
def dummy_processor(df: pd.DataFrame) -> pd.DataFrame:
    """A dummy processor that returns a DataFrame."""
    return df


@log_processor
def dummy_non_df_return_processor(
    df: pd.DataFrame,  # pylint: disable=unused-argument
) -> str:
    """Returns a string instead of a DataFrame."""
    return "not a dataframe"


def test_log_processor_success():
    """Test that the log_processor decorator logs the function name, input and output."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    with patch("data_cleaning_framework.clean_data.logger") as mock_logger:
        result = dummy_processor(df)

        assert result is df
        mock_logger.info.assert_called_once_with(
            "Function %s resulted in DataFrame with shape [%d, %d]. "
            "Args: %s, kwargs: %s",
            "dummy_processor",
            3,
            2,
            "<DataFrame>",
            "",
        )


def test_log_processor_non_df_return():
    """
    Test that the log_processor decorator logs an error
    if the function does not return a DataFrame.
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    with patch("data_cleaning_framework.clean_data.logger") as mock_logger:
        with pytest.raises(AssertionError, match="must return a DataFrame"):
            dummy_non_df_return_processor(df)

        mock_logger.error.assert_called_once()
        assert "Function dummy_non_df_return_processor must return a DataFrame" in str(
            mock_logger.error.call_args
        )


def test_log_processor_exception_handling():
    """Test that the log_processor decorator logs an exception if one occurs."""

    @log_processor
    def dummy_processor_with_exception(df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("An error occurred")

    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    with patch("data_cleaning_framework.clean_data.logger") as mock_logger:
        with pytest.raises(ValueError, match="An error occurred"):
            dummy_processor_with_exception(df)

        mock_logger.error.assert_called_once_with(
            "Error while calling function %s: %s",
            "dummy_processor_with_exception",
            "ValueError('An error occurred')",
        )
