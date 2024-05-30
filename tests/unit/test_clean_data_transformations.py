"""Tests the data transformation functions in the clean_data module"""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np
import pandas as pd
from data_cleaning_framework.clean_data import (
    assign_constant_columns,
    rename_columns,
    drop_rows,
    add_missing_columns,
    replace_values,
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
