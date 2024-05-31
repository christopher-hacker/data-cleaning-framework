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
    process_single_file,
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

    with mock.patch(
        "data_cleaning_framework.clean_data.get_cleaners",
        return_value=[(cleaner_func, args)],
    ):
        result = apply_cleaners(sample_df, schema_columns)
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

    with mock.patch(
        "data_cleaning_framework.clean_data.get_cleaners",
        return_value=[(cleaner_func, args)],
    ):
        expected = sample_df.copy()
        result = apply_cleaners(sample_df, schema_columns)
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

    with mock.patch(
        "data_cleaning_framework.clean_data.get_cleaners",
        return_value=[(cleaner_func, args)],
    ):
        expected = sample_df.copy()
        result = apply_cleaners(sample_df, schema_columns)
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

    with mock.patch(
        "data_cleaning_framework.clean_data.get_cleaners",
        return_value=[(cleaner_func, args)],
    ):
        with pytest.raises(
            ValueError, match="Cleaner cleaner_func was not applied to any columns."
        ):
            apply_cleaners(sample_df, schema_columns)


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
        schema=None,
    )


@pytest.fixture
def data_config():
    """Fixture that returns a data config."""
    return DataConfig(
        output_file="output.csv",
        assign_constant_columns=None,
        input_files=[],
        schema_file=None,
        cleaners_file=None,
    )


@pytest.fixture
def schema_model():
    class MockSchema(pa.SchemaModel):
        col1: pa.typing.Series[int]
        col2: pa.typing.Series[int]
        col3: pa.typing.Series[str]

        @classmethod
        def to_schema(cls):
            return cls.to_schema()

    return MockSchema


def test_process_single_file(
    sample_df, input_file_config, data_config, valid_columns, schema_model
):
    """Test process_single_file function."""
    with mock.patch(
        "data_cleaning_framework.clean_data.load_data", return_value=sample_df
    ) as mock_load_data, mock.patch(
        "data_cleaning_framework.clean_data.drop_rows", return_value=sample_df
    ) as mock_drop_rows, mock.patch(
        "data_cleaning_framework.clean_data.rename_columns", return_value=sample_df
    ) as mock_rename_columns, mock.patch(
        "data_cleaning_framework.clean_data.apply_query", return_value=sample_df
    ) as mock_apply_query, mock.patch(
        "data_cleaning_framework.clean_data.assign_constant_columns",
        return_value=sample_df,
    ) as mock_assign_constant_columns, mock.patch(
        "data_cleaning_framework.clean_data.replace_values", return_value=sample_df
    ) as mock_replace_values, mock.patch(
        "data_cleaning_framework.clean_data.add_missing_columns", return_value=sample_df
    ) as mock_add_missing_columns, mock.patch(
        "data_cleaning_framework.clean_data.apply_cleaners", return_value=sample_df
    ) as mock_apply_cleaners:

        with mock.patch.object(
            schema_model,
            "to_schema",
            return_value=mock.Mock(validate=mock.Mock(return_value=sample_df)),
        ):
            result = process_single_file(
                input_file_config, data_config, valid_columns, schema_model
            )

            mock_load_data.assert_called_once_with(
                input_file_config.input_file, input_file_config
            )
            mock_drop_rows.assert_called_once_with(
                sample_df, input_file_config.drop_rows
            )
            mock_rename_columns.assert_called_once_with(
                sample_df,
                valid_columns=valid_columns,
                columns=input_file_config.rename_columns,
            )
            mock_apply_query.assert_called_once_with(sample_df, input_file_config.query)
            mock_assign_constant_columns.assert_called_once_with(
                sample_df,
                *(
                    arg
                    for arg in [
                        data_config.assign_constant_columns,
                        input_file_config.assign_constant_columns,
                    ]
                    if arg is not None
                )
            )
            mock_replace_values.assert_called_once_with(
                sample_df, input_file_config.replace_values
            )
            mock_add_missing_columns.assert_called_once_with(
                sample_df, valid_columns=valid_columns
            )
            mock_apply_cleaners.assert_called_once_with(sample_df, scenario=None)

            # Validate that the schema's validate method was called
            schema_model.to_schema().validate.assert_called_once_with(sample_df)

            pd.testing.assert_frame_equal(result, sample_df)
