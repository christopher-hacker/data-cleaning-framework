"""Cleaners to be applied to a single file."""

from data_cleaning_framework import cleaner


@cleaner(dataframe_wise=True)
def fill_na_with_something(df):
    """Fill NA values with a constant."""
    return df.fillna("something")
