"""Reads and writes data."""

from pathlib import Path
import importlib.util
import inspect
from typing import Optional, Union, Dict, Any, List, AnyStr
import geopandas as gpd
import pandas as pd
from pydantic import validate_call
import yaml
from .constants import SUPPORTED_OUTPUT_TYPES
from .models import DataConfig, InputFileConfig


def get_cleaners(
    module: Any,
) -> List[tuple]:
    """Returns a list of cleaner functions and their arguments sorted by their order."""
    cleaners = []
    # get all the functions in this module that have been decorated with the cleaner decorator
    for _, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and hasattr(obj, "is_cleaner"):
            cleaners.append(obj)

    # sort the functions by their order
    cleaners = sorted(cleaners, key=lambda x: getattr(x, "order", float("inf")))

    # Return the sorted functions along with their cleaner_args
    return [(func, func.cleaner_args) for func in cleaners]


@validate_call
def import_module_from_path(module_path: AnyStr) -> Any:
    """Imports a module from a file and inserts it into the global namespace."""
    spec = importlib.util.spec_from_file_location("module.name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@validate_call
def load_user_modules(
    schema_file: str,
    cleaners_files: Optional[List[str]] = None,
) -> tuple:
    """Loads the necessary objects from user-defined modules."""
    schema_module = import_module_from_path(schema_file)

    try:
        schema = schema_module.Schema
    except AttributeError as exc:
        raise ValueError(
            "The schema file must contain a class called 'Schema' that "
            "inherits from pandera.SchemaModel"
        ) from exc

    if cleaners_files is not None:
        cleaners = []
        for cleaners_file in cleaners_files:
            cleaners_module = import_module_from_path(cleaners_file)
            cleaners.extend(get_cleaners(cleaners_module))
    else:
        cleaners = []

    return schema, cleaners


def get_args(config_file: str) -> DataConfig:
    """Gets arguments from config.yml."""
    with open(config_file, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    if isinstance(config, dict):
        return DataConfig(**config)
    if isinstance(config, list):
        return [DataConfig(**c) for c in config]

    raise ValueError(f"Invalid config file: {config_file}")


def read_excel_file(
    filename: str, sheet_name: str, skip_rows: Optional[int] = None, **kwargs
) -> pd.DataFrame:
    """Reads an Excel file."""
    try:
        df = pd.read_excel(
            filename, sheet_name=sheet_name, skiprows=skip_rows, **kwargs
        )
    except ValueError as exc:
        # gives you better error messages than the default pandas behavior, which
        # doesn't tell you which sheets are available
        existing_sheets = pd.ExcelFile(filename).sheet_names
        if sheet_name not in existing_sheets:
            raise ValueError(
                f"Error while reading {filename}. Sheet '{sheet_name}' not found. "
                f"Existing sheets: {existing_sheets}"
            ) from exc

    return df


def read_file(
    filename: str,
    sheet_name: Union[str, int] = 0,
    skip_rows: Optional[int] = None,
    xy_columns: Optional[Dict[str, str]] = None,
    crs: Optional[str] = None,
) -> pd.DataFrame:
    """Reads a file."""
    df = None
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = read_excel_file(filename, sheet_name=sheet_name, skiprows=skip_rows)
    if filename.endswith(".csv") or filename.endswith(".csv.gz"):
        df = pd.read_csv(
            filename,
            skiprows=skip_rows,
            compression="gzip" if filename.endswith(".gz") else None,
        )
    if filename.endswith(".geojson") or filename.endswith(".shp"):
        # the default crs is None, which will default to EPSG:4326
        df = gpd.read_file(filename, crs=crs)

    # convert to geospatial dataframe if necessary
    if not isinstance(df, gpd.GeoDataFrame) and xy_columns is not None:
        assert (
            crs is not None
        ), "You must explicitly provide a CRS when using xy_columns."
        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[xy_columns["x"]], df[xy_columns["y"]]),
            crs=crs,
        )
        # delete the original x and y columns
        df.drop(columns=[xy_columns["x"], xy_columns["y"]], inplace=True)

    if df is not None:
        return df

    raise ValueError(f"Unsupported file type: {filename}")


def call_preprocess_from_file(
    relative_path: str,
    kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Calls the function 'preprocess' from a provided python file."""
    module = import_module_from_path(relative_path)

    if hasattr(module, "preprocess"):
        if kwargs is None:
            return module.preprocess()

        return module.preprocess(**kwargs)

    raise ValueError(
        f"The function 'preprocess' was not found in the given file '{relative_path}'"
    )


@validate_call
def load_data(
    input_file_config: InputFileConfig,
    logger: Any,
) -> pd.DataFrame:
    """Reads an input file, or gets the output of a preprocessor function."""
    if input_file_config.preprocessor is not None:
        logger.info(
            f"Using preprocessor: {input_file_config.preprocessor.path} and "
            f"kwargs: {input_file_config.preprocessor.kwargs}"
        )
        df = call_preprocess_from_file(
            input_file_config.preprocessor.path,
            input_file_config.preprocessor.kwargs,
        )
    else:
        logger.info(f"Reading file: {input_file_config.input_file}")
        df = read_file(
            input_file_config.input_file,
            input_file_config.sheet_name,
            input_file_config.skip_rows,
            input_file_config.xy_columns,
            input_file_config.crs,
        )

    if input_file_config.to_crs is not None and isinstance(df, gpd.GeoDataFrame):
        logger.info(f"Converting CRS to {input_file_config.to_crs}")
        df = df.to_crs(input_file_config.to_crs)

    logger.info(
        f"Read {len(df)} rows and {len(df.columns)} columns from {input_file_config.input_file}. "
        f"Columns are: {df.columns}"
    )
    return df


@validate_call
def write_data(
    df: Any,
    output_file: str,
    logger: Any,
    append: bool = False,
) -> None:
    """Writes a DataFrame to a file."""
    logger.info(f"Attempting to write DataFrame to {output_file}")
    # get all file extensions, including if there are multiple like .csv.gz
    file_extension = "".join(Path(output_file).suffixes)
    if file_extension in SUPPORTED_OUTPUT_TYPES["csv"]:
        df.to_csv(
            output_file,
            index=False,
            compression="gzip" if output_file.endswith(".gz") else None,
            mode="a" if append else "w",
            header=not append,
        )
    elif file_extension in SUPPORTED_OUTPUT_TYPES["pkl"]:
        df.to_pickle(output_file)
    elif file_extension in SUPPORTED_OUTPUT_TYPES["geojson"]:
        df.to_file(output_file, driver="GeoJSON")
    else:
        raise NotImplementedError(f"Unsupported file type: {output_file}. ")

    logger.info(f"Successfully wrote DataFrame to {output_file}")
