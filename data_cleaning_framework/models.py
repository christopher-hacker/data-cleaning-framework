"""Contains Pydantic models for the data cleaning framework."""

# pylint: disable=too-few-public-methods

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, model_validator, field_validator
from .constants import SUPPORTED_OUTPUT_TYPES


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

    @model_validator(mode="before")
    @classmethod
    def validate_geospatial_values(cls, values):
        """Validates the geospatial values provided."""
        # check that if xy columns are provided, crs is also provided
        if values.get("xy_columns") and not values.get("crs"):
            raise ValueError("If xy_columns are provided, crs must also be provided.")
        # that that if xy columns are provided, they are dictionaries with only x and y keys
        if values.get("xy_columns"):
            if not isinstance(values["xy_columns"], dict):
                raise ValueError("xy_columns must be a dictionary.")
            assert sorted(list(values["xy_columns"].keys())) == [
                "x",
                "y",
            ], f"xy_columns must have keys 'x' and 'y'. Got {values['xy_columns']}"
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
    drop_columns: Optional[List[Union[str, int]]] = Field(
        default=None,
        description="List of column names or indices to drop from the DataFrame ",
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
    drop_extra_columns: Optional[bool] = Field(
        default=False,
        description="Whether to drop columns with all missing values. Defaults to False.",
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
    xy_columns: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary of column names containing latitude and longitude."
        " If provided, the dataframe will be converted to a GeoDataFrame by passing those columns"
        " to the geopandas.points_from_xy function.",
    )
    crs: Optional[str] = Field(
        default=None,
        description="Coordinate reference system to use when creating a GeoDataFrame."
        " If provided, the GeoDataFrame will be set to this CRS.",
    )
    cleaners_files: Optional[List[str]] = Field(
        default=None,
        description="A list of the paths to cleaners files to apply only to this file. "
        "Defaults to None.",
    )
    to_crs: Optional[str] = Field(
        default=None,
        description="Coordinate reference system to convert to, if working with a GeoDataFrame.",
    )
    read_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of keyword arguments to pass to the read function."
        " For example, you can pass the 'dtype' argument to specify column data types.",
    )


class DataConfig(BaseModel):
    """Configuration details for a task."""

    class Config:
        """Pydantic configuration options."""

        extra = "forbid"

    @field_validator("output_file")
    def validate_output_file(  # pylint: disable=no-self-argument
        cls: Any,
        value: str,
    ) -> str:
        """Validates that the output file has a supported extension."""
        extension_valid = False
        for allowed_extensions in SUPPORTED_OUTPUT_TYPES.values():
            for allowed_extension in allowed_extensions:
                if value.endswith(allowed_extension):
                    extension_valid = True
                    break
        if not extension_valid:
            extension = "".join(Path(value).suffixes)
            raise NotImplementedError(
                f"Output file '{value}' has an unsupported extension '{extension}'"
            )
        return value

    def list_referenced_files(self):
        """Lists all files referenced in the configuration."""
        files = []
        for input_file in self.input_files:  # pylint: disable=not-an-iterable
            if input_file.input_file:
                files.append(input_file.input_file)
            if input_file.preprocessor:
                files.append(input_file.preprocessor.path)
            if input_file.cleaners_files:
                files.extend(input_file.cleaners_files)
        if self.schema_file:
            files.append(self.schema_file)
        if self.cleaners_files:
            files.extend(self.cleaners_files)
        return files

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
