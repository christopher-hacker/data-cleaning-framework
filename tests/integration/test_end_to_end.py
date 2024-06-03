"""Simplest end-to-end test for the framework."""

from data_cleaning_framework.clean_data import main


def test_simple():
    """Test the simplest end-to-end test for the framework."""
    main(config_file="tests/data/simple-config.yaml")
