"""Tests the data transformation functions in the clean_data module"""

# pylint: disable=redefined-outer-name

from unittest import mock
import pytest
import numpy as np
import pandas as pd
import pandera as pa
from data_cleaning_framework.models import InputFileConfig, DataConfig
from data_cleaning_framework.clean_data import (
    assign_constant_columns,
    rename_columns,
    drop_rows,
    add_missing_columns,
    replace_values,
    apply_query,
    apply_cleaners,
    process_and_write_file,
    CleaningFailedError,
)


@pytest.fixture
def sample_df():
    """Fixture that returns a basic dataframe"""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})


@pytest.fixture
def valid_columns():
    """Fixture that returns a list of valid columns"""
    return ["new_col1", "new_col2"]


@pytest.fixture
def sample_valid_df():
    """Fixture that returns a basic dataframe"""
    return pd.DataFrame({"new_col1": [1, 2, 3], "new_col2": [4, 5, 6]})


def test_assign_constant_columns(sample_df):
    """Test the assign_constant_columns function"""
    result = assign_constant_columns(sample_df, {"new_col1": "new_val1"})
    assert result.columns.tolist() == ["col1", "col2", "new_col1"]
    assert result["new_col1"].tolist() == ["new_val1", "new_val1", "new_val1"]


def test_rename_columns_no_columns(sample_df, valid_columns):
    """Test rename_columns with no columns provided."""
    result = rename_columns(sample_df, valid_columns=valid_columns)
    pd.testing.assert_frame_equal(result, sample_df)


def test_rename_columns_by_name(sample_df, valid_columns):
    """Test rename_columns by column names."""
    columns = {"col1": "new_col1", "col2": "new_col2"}
    result = rename_columns(sample_df, valid_columns, columns)
    expected = pd.DataFrame({"new_col1": [1, 2, 3], "new_col2": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_rename_columns_by_index(sample_df, valid_columns):
    """Test rename_columns by column index."""
    columns = {0: "new_col1", 1: "new_col2"}
    result = rename_columns(sample_df, valid_columns, columns)
    expected = pd.DataFrame({"new_col1": [1, 2, 3], "new_col2": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_rename_columns_by_index_out_of_range(sample_df, valid_columns):
    """Test rename_columns by column index with out of range index."""
    columns = {0: "new_col1", 2: "new_col2"}
    with pytest.raises(ValueError):
        rename_columns(sample_df, valid_columns, columns)


def test_rename_columns_by_name_invalid_name(sample_df, valid_columns):
    """Test rename_columns by column names with invalid name."""
    columns = {"col1": "new_col1", "col3": "new_col2"}
    with pytest.raises(ValueError):
        rename_columns(sample_df, valid_columns, columns)


def test_rename_columns_by_name_existing_name(sample_df, valid_columns):
    """Test rename_columns by column names with existing name."""
    columns = {"col1": "col2", "col2": "new_col2"}
    with pytest.raises(ValueError):
        rename_columns(sample_df, valid_columns, columns)


def test_rename_columns_by_name_not_in_schema(sample_df, valid_columns):
    """Test rename_columns by column names with invalid name."""
    columns = {"col1": "new_col1", "col2": "new_col3"}
    with pytest.raises(ValueError):
        rename_columns(sample_df, valid_columns, columns)


def test_drop_rows_no_drop(sample_df):
    """Test drop_rows with no rows to drop."""
    result = drop_rows(sample_df)
    pd.testing.assert_frame_equal(result, sample_df)


def test_drop_rows(sample_df):
    """Test drop_rows with rows to drop."""
    result = drop_rows(sample_df, rows=[0, 2])
    expected = pd.DataFrame({"col1": [2], "col2": [5]})
    pd.testing.assert_frame_equal(result, expected)


def test_add_missing_columns(sample_valid_df, valid_columns):
    """Test add_missing_columns with missing columns."""
    # drop the last col from the valid df
    sample_valid_df = sample_valid_df.drop(columns=["new_col2"])
    result = add_missing_columns(sample_valid_df, valid_columns)
    expected = pd.DataFrame(
        {"new_col1": [1, 2, 3], "new_col2": [np.NaN, np.NaN, np.NaN]}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_add_missing_columns_no_add(sample_valid_df, valid_columns):
    """Test add_missing_columns with no missing columns."""
    result = add_missing_columns(sample_valid_df, valid_columns)
    pd.testing.assert_frame_equal(result, sample_valid_df)


def test_replace_values(sample_df):
    """Test the replace_values function"""
    sample_df.loc[1, "col1"] = "missing"
    result = replace_values(sample_df, {"col1": {"missing": 0}})
    assert result["col1"].tolist() == [1, 0, 3]


def test_apply_query(sample_df):
    """Test the apply_query function"""
    query = "col1 == 1"
    result = apply_query(sample_df, query)
    assert result["col1"].tolist() == [1]


def test_apply_query_no_query(sample_df):
    """Test the apply_query function with no query"""
    result = apply_query(sample_df, None)
    pd.testing.assert_frame_equal(result, sample_df)


def test_apply_query_undefined_variable(sample_df):
    """Test the apply_query function with an undefined variable"""
    query = "col1 == x"
    with pytest.raises(ValueError):
        apply_query(sample_df, query)


@pytest.fixture
def schema_columns():
    """Fixture that returns a schema with columns"""
    return {"col1": pa.Column(int), "col2": pa.Column(int)}


def test_apply_cleaners_dataframe_wise(sample_df, schema_columns):
    """Test apply_cleaners with a dataframe-wise cleaner."""

    def cleaner_func(df):
        return df[df["col1"] > 1]

    cleaner_func.func_name = "cleaner_func"

    args = mock.Mock()
    args.dataframe_wise = True
    args.columns = None
    args.dtypes = None

    result = apply_cleaners(sample_df, [(cleaner_func, args)], schema_columns)
    expected = sample_df[sample_df["col1"] > 1]
    pd.testing.assert_frame_equal(result, expected)


def test_apply_cleaners_column_wise(sample_df, schema_columns):
    """Test apply_cleaners with a column-wise cleaner."""

    def cleaner_func(x):
        return x * 2

    cleaner_func.func_name = "cleaner_func"

    args = mock.Mock()
    args.dataframe_wise = False
    args.columns = ["col1"]
    args.dtypes = None

    expected = sample_df.copy()
    result = apply_cleaners(sample_df, [(cleaner_func, args)], schema_columns)
    expected["col1"] = expected["col1"].apply(cleaner_func)
    pd.testing.assert_frame_equal(result, expected)


def test_apply_cleaners_dtype_wise(sample_df, schema_columns):
    """Test apply_cleaners with a dtype-wise cleaner."""

    def cleaner_func(x):
        return x * 2

    cleaner_func.func_name = "cleaner_func"

    args = mock.Mock()
    args.dataframe_wise = False
    args.columns = None
    args.dtypes = [int]

    expected = sample_df.copy()
    result = apply_cleaners(sample_df, [(cleaner_func, args)], schema_columns)
    expected["col1"] = expected["col1"].apply(cleaner_func)
    expected["col2"] = expected["col2"].apply(cleaner_func)
    pd.testing.assert_frame_equal(result, expected)


def test_apply_cleaners_no_cleaner_called(sample_df, schema_columns):
    """Test apply_cleaners when no cleaner is called."""

    def cleaner_func(df):
        return df

    cleaner_func.func_name = "cleaner_func"

    args = mock.Mock()
    args.dataframe_wise = False
    args.columns = None
    args.dtypes = None

    with pytest.raises(
        ValueError, match="Cleaner cleaner_func was not applied to any columns."
    ):
        apply_cleaners(sample_df, [(cleaner_func, args)], schema_columns)


@pytest.fixture
def input_file_config():
    """Fixture that returns an input file config."""
    return InputFileConfig(
        input_file="dummy_file.csv",
        drop_rows=None,
        rename_columns=None,
        query=None,
        assign_constant_columns=None,
        replace_values=None,
    )


@pytest.fixture
def data_config():
    """Fixture that returns a data config."""
    return DataConfig(
        output_file="output.csv",
        assign_constant_columns=None,
        input_files=[],
        schema_file="tests/data/test_schema.py",
        cleaners_files=None,
    )


@pytest.fixture
def schema_model():
    """Fixture that returns a mock schema model."""

    class MockSchema(pa.SchemaModel):
        """Mock schema model."""

        col1: pa.typing.Series[int]
        col2: pa.typing.Series[int]
        col3: pa.typing.Series[str]

        @classmethod
        def to_schema(cls):
            return cls.to_schema()

    return MockSchema


def test_process_and_write_file_success(
    sample_df, input_file_config, data_config, schema_model
):
    """Test process_and_write_file function with successful processing and writing."""
    with mock.patch(
        "data_cleaning_framework.clean_data.process_single_file", return_value=sample_df
    ) as mock_process_single_file, mock.patch("pandas.DataFrame.to_csv") as mock_to_csv:

        process_and_write_file(
            input_file_config=input_file_config,
            yaml_args=data_config,
            schema=schema_model,
            valid_columns=["col1", "col2"],
            cleaners=None,
        )

        mock_process_single_file.assert_called_once_with(
            input_file_config=input_file_config,
            args=data_config,
            valid_columns=["col1", "col2"],
            schema=schema_model,
            cleaners=None,
        )
        mock_to_csv.assert_called_once_with(
            data_config.output_file, index=False, mode="a", header=False
        )


def test_process_and_write_file_with_exception(
    input_file_config, data_config, schema_model
):
    """Test process_and_write_file function when an exception occurs during processing."""
    with mock.patch(
        "data_cleaning_framework.clean_data.process_single_file",
        side_effect=Exception("Processing error"),
    ), pytest.raises(CleaningFailedError) as exc_info:

        process_and_write_file(
            input_file_config=input_file_config,
            yaml_args=data_config,
            schema=schema_model,
            valid_columns=["col1", "col2"],
        )

        assert "Error while cleaning file." in str(exc_info.value)
        assert f"Please check the file {input_file_config.input_file}." in str(
            exc_info.value
        )


def test_process_and_write_file_with_preprocessor_exception(
    input_file_config, data_config, schema_model
):
    """Test process_and_write_file function when an exception occurs with preprocessor."""
    input_file_config.preprocessor = mock.Mock()
    input_file_config.preprocessor.path = "dummy_path.py"
    input_file_config.preprocessor.kwargs = {"param1": "value1"}

    with mock.patch(
        "data_cleaning_framework.clean_data.process_single_file",
        side_effect=Exception("Processing error"),
    ), pytest.raises(CleaningFailedError) as exc_info:

        process_and_write_file(
            input_file_config=input_file_config,
            yaml_args=data_config,
            schema=schema_model,
            valid_columns=["col1", "col2"],
        )

        assert "Error while cleaning file." in str(exc_info.value)
        assert (
            f"Please check the preprocess function in {input_file_config.preprocessor.path}."
            in str(exc_info.value)
        )
        assert f"Args: {input_file_config.preprocessor.kwargs}." in str(exc_info.value)


@pytest.fixture
def yaml_args():
    """Fixture that returns a mock yaml args."""
    return mock.Mock(
        input_files=["input1.csv", "input2.csv"],
        output_file="output.csv",
        schema_file=None,
        cleaners_file=None,
    )
