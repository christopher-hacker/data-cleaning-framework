"""Data i/o tests for the clean_data module."""

import builtins
import pytest
from data_cleaning_framework.clean_data import insert_into_namespace


@pytest.fixture(autouse=True)
def cleanup_namespace():
    """Fixture to clean up the namespace after each test"""
    yield
    # Teardown code: Remove 'cleaners' and 'dummy_cleaner' from builtins if they exist
    for var in ["cleaners", "dummy_cleaner"]:
        try:
            delattr(builtins, var)
        except AttributeError:
            pass


def test_insert_into_namespace():
    """tests the insert_into_namespace function"""
    # to start, the namespace should not contain the module 'cleaners'
    with pytest.raises(NameError):
        cleaners  # pylint: disable=pointless-statement
    insert_into_namespace("tests/unit/data/cleaners.py", "cleaners")
    assert cleaners


def test_insert_into_namespace_error():
    """tests the insert_into_namespace function when the module does not exist"""
    with pytest.raises(FileNotFoundError):
        insert_into_namespace("tests/unit/data/cleaners2.py", "cleaners")
    with pytest.raises(NameError):
        cleaners  # pylint: disable=pointless-statement


def test_insert_into_namespace_object_names():
    """tests the insert_into_namespace function when the object names are provided"""
    # to start, the namespace should not contain the module 'cleaners'
    with pytest.raises(NameError):
        cleaners  # pylint: disable=pointless-statement
    insert_into_namespace("tests/unit/data/cleaners.py", "cleaners", "dummy_cleaner")
    assert dummy_cleaner  # pylint: disable=undefined-variable
