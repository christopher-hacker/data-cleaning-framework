"""Data i/o tests for the clean_data module."""

# pylint: disable=pointless-statement,undefined-variable,redefined-outer-name

import builtins
from unittest import mock
import pandas as pd
import pandera as pa
import pytest
from data_cleaning_framework.io import (
    import_module_from_path,
    load_user_modules,
    get_args,
    read_excel_file,
    read_file,
    call_preprocess_from_file,
    load_data,
)
from data_cleaning_framework.models import (
    DataConfig,
    InputFileConfig,
    PreProcessorConfig,
)


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


def test_import_module_from_path():
    """tests the import_module_from_path function"""
    cleaners = import_module_from_path("tests/data/simple_cleaners.py")
    assert cleaners


def test_import_module_from_path_error():
    """tests the import_module_from_path function when the module does not exist"""
    with pytest.raises(FileNotFoundError):
        import_module_from_path("tests/data/nonexistentcleaners.py")


def test_load_user_modules():
    """tests the load_user_modules function"""
    schema, cleaners = load_user_modules(
        "tests/data/simple_schema.py", ["tests/data/simple_cleaners.py"]
    )
    assert isinstance(schema, pa.api.base.model.MetaModel), type(schema)
    assert cleaners


def test_load_user_modules_missing_schema():
    """tests the load_user_modules function when the schema is missing"""
    with pytest.raises(ValueError, match="The schema file must contain a class called"):
        load_user_modules(
            "tests/data/schema_broken.py", ["tests/data/simple_cleaners.py"]
        )


def test_load_user_modules_no_cleaners():
    """Tests the load_user_modules function when the cleaners module is missing"""
    schema, cleaners = load_user_modules("tests/data/simple_schema.py", None)
    assert isinstance(schema, pa.api.base.model.MetaModel), type(schema)
    assert not cleaners


def test_get_args():
    """Tests the get_args function"""
    config_path = "tests/data/simple-config.yaml"
    args = get_args(config_path)
    assert isinstance(args, DataConfig)


def test_get_args_with_list_config():
    """Tests the get_args function with a list configuration"""
    config_path = "tests/data/list-config.yaml"
    args = get_args(config_path)
    assert isinstance(args, list)
    assert all(isinstance(arg, DataConfig) for arg in args)


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
            filename, sheet_name=sheet_name, skiprows=skip_rows
        )
        pd.testing.assert_frame_equal(result, mock_df)


def test_read_file_csv():
    """Test reading a CSV file."""
    filename = "test.csv"
    skip_rows = 1
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with mock.patch("pandas.read_csv", return_value=mock_df) as mock_read_csv:
        result = read_file(filename, skip_rows=skip_rows)
        mock_read_csv.assert_called_once_with(
            filename, skiprows=skip_rows, compression=None
        )
        pd.testing.assert_frame_equal(result, mock_df)


def test_read_file_unsupported():
    """Test reading an unsupported file type."""
    filename = "test.txt"

    with pytest.raises(ValueError, match=f"Unsupported file type: {filename}"):
        read_file(filename)


@pytest.fixture
def mock_preprocess_df():
    """Fixture to create a mock DataFrame for preprocess functions."""
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


@pytest.fixture
def mock_read_file_df():
    """Fixture to create a mock DataFrame for read_file."""
    return pd.DataFrame({"col1": [7, 8, 9], "col2": [10, 11, 12]})


def test_load_data_with_preprocessor(mock_preprocess_df):
    """Test load_data with a preprocessor."""
    preprocessor_config = PreProcessorConfig(
        path="dummy_path.py", kwargs={"param1": "value1"}
    )

    input_file_config = InputFileConfig(
        preprocessor=preprocessor_config,
        sheet_name=None,
        skip_rows=None,
    )

    with mock.patch(
        "data_cleaning_framework.io.call_preprocess_from_file",
        return_value=mock_preprocess_df,
    ) as mock_call_preprocess:
        result = load_data(input_file_config, logger=mock.Mock())
        mock_call_preprocess.assert_called_once_with(
            preprocessor_config.path, preprocessor_config.kwargs
        )
        pd.testing.assert_frame_equal(result, mock_preprocess_df)


def test_load_data_with_file(mock_read_file_df):
    """Test load_data with a file."""
    input_file = "dummy_file.csv"
    input_file_config = InputFileConfig(
        input_file=input_file,
        sheet_name="Sheet1",
        skip_rows=1,
    )

    with mock.patch(
        "data_cleaning_framework.io.read_file", return_value=mock_read_file_df
    ) as mock_read_file:
        result = load_data(input_file_config, logger=mock.Mock())
        mock_read_file.assert_called_once_with(
            input_file,
            input_file_config.sheet_name,
            input_file_config.skip_rows,
            None,
            None,
        )
        pd.testing.assert_frame_equal(result, mock_read_file_df)
