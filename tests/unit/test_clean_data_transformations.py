"""Tests the data transformation functions in the clean_data module"""

import pandas as pd
from data_cleaning_framework.clean_data import assign_constant_columns


def test_assign_constant_columns():
    """Test the assign_constant_columns function"""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    result = assign_constant_columns(df, {"new_col1": "new_val1"})
    assert result.columns.tolist() == ["col1", "col2", "new_col1"]
    assert result["new_col1"].tolist() == ["new_val1", "new_val1", "new_val1"]
