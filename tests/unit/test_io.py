"""Data i/o tests for the clean_data module."""

# pylint: disable=pointless-statement,undefined-variable,redefined-outer-name

import builtins
from unittest import mock
import pandas as pd
import pytest
from data_cleaning_framework.io import (
    insert_into_namespace,
    load_user_modules,
    get_args,
    read_excel_file,
    read_file,
    call_preprocess_from_file,
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

    with mock.patch("data_cleaning_framework.io.insert_into_namespace") as mock_insert:
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

    with mock.patch("data_cleaning_framework.io.insert_into_namespace") as mock_insert:
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


def test_read_excel_file():
    """Tests the read_excel_file function"""
    df = read_excel_file("tests/data/simple-input.xlsx", sheet_name="Sheet1")
    assert isinstance(df, pd.DataFrame)


def test_read_excel_file_missing_sheet():
    """Tests the read_excel_file function when the sheet is missing"""
    # check that the error message raised contains the text "Existing sheets:"
    with pytest.raises(ValueError, match="Existing sheets:"):
        read_excel_file("tests/data/simple-input.xlsx", sheet_name="Sheet2")


def test_read_file_excel():
    """Test reading an Excel file."""
    filename = "test.xlsx"
    sheet_name = "Sheet1"
    skip_rows = 1
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with mock.patch(
        "data_cleaning_framework.io.read_excel_file", return_value=mock_df
    ) as mock_read_excel:
        result = read_file(filename, sheet_name, skip_rows)
        mock_read_excel.assert_called_once_with(
            filename, sheet_name=sheet_name, skip_rows=skip_rows
        )
        pd.testing.assert_frame_equal(result, mock_df)


def test_read_file_csv():
    """Test reading a CSV file."""
    filename = "test.csv"
    skip_rows = 1
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with mock.patch("pandas.read_csv", return_value=mock_df) as mock_read_csv:
        result = read_file(filename, skip_rows=skip_rows)
        mock_read_csv.assert_called_once_with(filename, skip_rows=skip_rows)
        pd.testing.assert_frame_equal(result, mock_df)


def test_read_file_unsupported():
    """Test reading an unsupported file type."""
    filename = "test.txt"

    with pytest.raises(ValueError, match=f"Unsupported file type: {filename}"):
        read_file(filename)


@pytest.fixture
def mock_preprocess_df():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})


def test_call_preprocess_from_file():
    """Test call_preprocess_from_file."""
    module_path = "tests/data/preprocess.py"
    result = call_preprocess_from_file(module_path)
    assert isinstance(result, pd.DataFrame)


def test_call_preprocess_from_file_no_kwargs(mock_preprocess_df):
    """Test call_preprocess_from_file without kwargs."""
    with mock.patch(
        "importlib.util.spec_from_file_location"
    ) as mock_spec_from_file_location:
        with mock.patch("importlib.util.module_from_spec") as mock_module_from_spec:
            mock_spec = mock.Mock()
            mock_spec.loader = mock.Mock()
            mock_module = mock.Mock()
            mock_spec_from_file_location.return_value = mock_spec
            mock_module_from_spec.return_value = mock_module
            mock_module.preprocess.return_value = mock_preprocess_df

            result = call_preprocess_from_file("dummy_path.py")
            mock_spec.loader.exec_module.assert_called_once_with(mock_module)
            assert result.equals(mock_preprocess_df)


def test_call_preprocess_from_file_with_kwargs(mock_preprocess_df):
    """Test call_preprocess_from_file with kwargs."""
    with mock.patch(
        "importlib.util.spec_from_file_location"
    ) as mock_spec_from_file_location:
        with mock.patch("importlib.util.module_from_spec") as mock_module_from_spec:
            mock_spec = mock.Mock()
            mock_spec.loader = mock.Mock()
            mock_module = mock.Mock()
            mock_spec_from_file_location.return_value = mock_spec
            mock_module_from_spec.return_value = mock_module
            mock_module.preprocess.return_value = mock_preprocess_df

            kwargs = {"param1": "value1"}
            result = call_preprocess_from_file("dummy_path.py", kwargs)
            mock_module.preprocess.assert_called_once_with(**kwargs)
            mock_spec.loader.exec_module.assert_called_once_with(mock_module)
            assert result.equals(mock_preprocess_df)


def test_call_preprocess_from_file_exec_module_error():
    """Test call_preprocess_from_file when exec_module raises an error."""
    with mock.patch(
        "importlib.util.spec_from_file_location"
    ) as mock_spec_from_file_location:
        with mock.patch("importlib.util.module_from_spec") as mock_module_from_spec:
            mock_spec = mock.Mock()
            mock_spec.loader = mock.Mock()
            mock_module = mock.Mock()
            mock_spec_from_file_location.return_value = mock_spec
            mock_module_from_spec.return_value = mock_module
            mock_spec.loader.exec_module.side_effect = ImportError

            with pytest.raises(ImportError):
                call_preprocess_from_file("dummy_path.py")


def test_call_preprocess_from_file_no_preprocess_function_in_module():
    """Test call_preprocess_from_file when the module does not have a preprocess function."""
    preprocess_path = "tests/data/preprocess_broken.py"
    with pytest.raises(ValueError):
        call_preprocess_from_file(preprocess_path)
