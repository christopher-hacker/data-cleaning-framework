"""Contains Pydantic models for the data cleaning framework."""

# pylint: disable=too-few-public-methods

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, model_validator


class PreProcessorConfig(BaseModel):
    """Configuration details for a preprocessor function."""

    class Config:
        """Pydantic configuration options."""

        extra = "forbid"

    path: str = Field(
        description="The path to a Python file containing a function called preprocess "
        "that accepts no arguments and returns a dataframe."
    )
    kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of keyword arguments to pass to the preprocess function.",
    )


class InputFileConfig(BaseModel):
    """Configuration details for an input file."""

    class Config:
        """Pydantic configuration options."""

        extra = "forbid"

    @model_validator(mode="before")
    @classmethod
    def validate_file_getter(cls, values):
        """Validates that either input_file or preprocessor is provided."""
        if "input_file" not in values and "preprocessor" not in values:
            raise ValueError("Either input_file or preprocessor must be provided.")
        if "input_file" in values and "preprocessor" in values:
            raise ValueError("Only one of input_file or preprocessor can be provided.")
        return values

    sheet_name: Optional[str] = Field(
        default=0,
        description="Name of the sheet to read from the Excel file. Defaults to the first sheet.",
    )
    input_file: Optional[str] = Field(
        default=None,
        description="Name of the input file to read from the input folder. Alternatively, "
        "you can provide a preprocessor function to read the file.",
    )
    preprocessor: Optional[PreProcessorConfig] = Field(
        default=None,
        description="The path to a Python file containing a function called "
        "preprocess that returns a dataframe."
        "If provided, this function will be called and the output will be "
        "used as the input to the cleaning process, instead of reading the file directly.",
    )
    skip_rows: Optional[int] = Field(
        default=None, description="Number of rows to skip when reading the Excel file."
    )
    drop_rows: Optional[List[int]] = Field(
        default=None,
        description="List of row indices to drop from the DataFrame "
        "after reading the Excel file.",
    )
    assign_constant_columns: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of column names to values to assign to all rows."
        "Similar to the assign_constant_columns field in DataConfig, "
        "but only applies to this file.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Query to apply to the DataFrame after reading the file. "
        "Query uses the names of the destination columns as specified in the "
        "columns field of the config.",
    )
    replace_values: Optional[Dict[str, Dict]] = Field(
        default=None,
        description="Dictionary of column names to dictionaries of values to replace.",
    )
    rename_columns: Optional[Dict[Union[int, str], str]] = Field(
        default=None,
        description="Dictionary of column names or indices to rename.",
    )
    date_columns: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary of column names to date formats to parse.",
    )
    date_errors: Optional[str] = Field(
        default="raise",
        description="How to handle date parsing errors. "
        "See pandas.to_datetime documentation for options.",
    )


class DataConfig(BaseModel):
    """Configuration details for a task."""

    class Config:
        """Pydantic configuration options."""

        extra = "forbid"

    output_file: str = Field(
        description="Name of the output file to write to the output folder."
    )
    schema_file: str = Field(description="The path to a schema file to use.")
    input_files: List[InputFileConfig] = Field(
        description="List of input file configuration details.",
    )
    assign_constant_columns: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of column names to values to assign to all rows.",
    )
    cleaners_files: Optional[List[str]] = Field(
        default=None,
        description="A list of the paths to cleaners files to use. Defaults to None. ",
    )


class CleanerArgs(BaseModel):
    """Arguments for a cleaner."""

    class Config:
        """Pydantic configuration options."""

        extra = "forbid"

    columns: Optional[List[str]] = None
    dtypes: Optional[List[type]] = None
    dataframe_wise: bool = False
    order: int = 0
    return_nulls: bool = False
