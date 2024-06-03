"""Tests the Click cli commands and related functions."""

from data_cleaning_framework.cli import reorder_keys


def test_reorder_keys_with_all_keys():
    """Test reorder_keys when all keys are present."""
    data = {
        "type": "example_type",
        "title": "example_title",
        "description": "example_description",
        "items": "example_items",
        "additional_key": "additional_value",
    }
    expected = {
        "title": "example_title",
        "type": "example_type",
        "description": "example_description",
        "items": "example_items",
        "additional_key": "additional_value",
    }
    assert reorder_keys(data) == expected


def test_reorder_keys_with_missing_keys():
    """Test reorder_keys when some keys are missing."""
    data = {
        "type": "example_type",
        "description": "example_description",
        "additional_key": "additional_value",
    }
    expected = {
        "type": "example_type",
        "description": "example_description",
        "additional_key": "additional_value",
    }
    assert reorder_keys(data) == expected


def test_reorder_keys_with_no_keys():
    """Test reorder_keys when no keys are present."""
    data = {
        "additional_key1": "additional_value1",
        "additional_key2": "additional_value2",
    }
    expected = {
        "additional_key1": "additional_value1",
        "additional_key2": "additional_value2",
    }
    assert reorder_keys(data) == expected


def test_reorder_keys_with_empty_dict():
    """Test reorder_keys with an empty dictionary."""
    data = {}
    expected = {}
    assert reorder_keys(data) == expected


def test_reorder_keys_with_extra_keys():
    """Test reorder_keys when extra keys are present."""
    data = {
        "title": "example_title",
        "extra_key1": "extra_value1",
        "extra_key2": "extra_value2",
        "type": "example_type",
        "description": "example_description",
    }
    expected = {
        "title": "example_title",
        "type": "example_type",
        "description": "example_description",
        "extra_key1": "extra_value1",
        "extra_key2": "extra_value2",
    }
    assert reorder_keys(data) == expected
