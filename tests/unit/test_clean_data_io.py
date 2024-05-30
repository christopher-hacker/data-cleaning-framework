"""Data i/o tests for the clean_data module."""

# pylint: disable=pointless-statement,undefined-variable

import builtins
from unittest import mock
import pytest
from data_cleaning_framework.clean_data import (
    insert_into_namespace,
    load_user_modules,
    get_args,
)
from data_cleaning_framework.models import DataConfig


@pytest.fixture(autouse=True)
def cleanup_namespace():
    """Fixture to clean up the namespace after each test"""
    yield
    # Teardown code: Remove items from builtins if they exist
    for var in ["cleaners", "dummy_cleaner", "Schema"]:
        try:
            delattr(builtins, var)
        except AttributeError:
            pass


def test_insert_into_namespace():
    """tests the insert_into_namespace function"""
    # to start, the namespace should not contain the module 'cleaners'
    with pytest.raises(NameError):
        cleaners
    insert_into_namespace("tests/data/cleaners.py", "cleaners")
    assert cleaners


def test_insert_into_namespace_error():
    """tests the insert_into_namespace function when the module does not exist"""
    with pytest.raises(FileNotFoundError):
        insert_into_namespace("tests/data/cleaners2.py", "cleaners")
    with pytest.raises(NameError):
        cleaners


def test_insert_into_namespace_object_names():
    """tests the insert_into_namespace function when the object names are provided"""
    # to start, the namespace should not contain the module 'cleaners'
    with pytest.raises(NameError):
        cleaners
    insert_into_namespace("tests/data/cleaners.py", "cleaners", "dummy_cleaner")
    assert dummy_cleaner  # pylint: disable=undefined-variable


def test_load_user_modules_from_path():
    """tests the load_user_modules_from_path function"""
    # to start, the namespace should not contain the modules 'cleaners' or 'schema'
    with pytest.raises(NameError):
        cleaners
    with pytest.raises(NameError):
        schema
    load_user_modules("tests/data/schema.py", "tests/data/cleaners.py")
    assert cleaners
    assert Schema


def test_load_user_modules_without_schema_file():
    """Test load_user_modules without schema_file"""
    cleaners_file = "tests/data/cleaners.py"

    with mock.patch(
        "data_cleaning_framework.clean_data.insert_into_namespace"
    ) as mock_insert:
        with mock.patch(
            "builtins.__import__",
            side_effect=lambda name, *args: (
                mock.Mock() if name == "schema" else __import__(name, *args)
            ),
        ):
            load_user_modules(cleaners_file=cleaners_file)
            mock_insert.assert_called_once_with(cleaners_file, "cleaners")


def test_load_user_modules_without_cleaners_file():
    """Test load_user_modules without cleaners_file"""
    schema_file = "tests/data/schema.py"

    with mock.patch(
        "data_cleaning_framework.clean_data.insert_into_namespace"
    ) as mock_insert:
        with mock.patch(
            "builtins.__import__",
            side_effect=lambda name, *args: (
                mock.Mock() if name == "cleaners" else __import__(name, *args)
            ),
        ):
            load_user_modules(schema_file=schema_file)
            mock_insert.assert_called_once_with(schema_file, "schema", "Schema")


def test_get_args():
    """Tests the get_args function"""
    config_path = "tests/data/simple-config.yaml"
    args = get_args(config_path)
    assert isinstance(args, DataConfig)
