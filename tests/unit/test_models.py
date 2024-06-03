"""Contains unit test for the models module."""

import pytest
from pydantic import ValidationError
from data_cleaning_framework.models import InputFileConfig, PreProcessorConfig


def test_input_file_config_validate_file_getter():
    """Test validate_file_getter method of InputFileConfig."""
    with pytest.raises(
        ValidationError, match="Either input_file or preprocessor must be provided."
    ):
        InputFileConfig()
    with pytest.raises(
        ValidationError, match="Only one of input_file or preprocessor can be provided."
    ):
        InputFileConfig(input_file="file", preprocessor="preprocessor")
    InputFileConfig(input_file="file")
    InputFileConfig(preprocessor=PreProcessorConfig(path="path"))
