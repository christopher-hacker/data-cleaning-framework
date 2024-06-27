"""Simplest end-to-end test for the framework."""

from pydantic import ValidationError
import pytest
from data_cleaning_framework.clean_data import main


def test_simple():
    """Test the simplest end-to-end test for the framework."""
    main(config_file="tests/data/simple-config.yaml")


def test_simple_invalid_argument():
    """Test the simplest end-to-end test for the framework."""
    with pytest.raises(ValidationError):
        main(config_file="tests/data/simple-config-invalid-arg.yaml")


def test_geo_with_geojson():
    """Test the simplest end-to-end test for the framework with geojson input data."""
    main(config_file="tests/data/geo-config.yaml")


def test_geo_with_csv():
    """Test the simplest end-to-end test for the framework with csv input data."""
    main(config_file="tests/data/geo-config-csv.yaml")
