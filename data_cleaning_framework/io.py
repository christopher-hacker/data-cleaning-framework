"""Reads and writes data."""

import importlib.util
import inspect
from typing import Optional, Union, Dict, Any, List, AnyStr
import pandas as pd
from pydantic import validate_call
import yaml
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


def load_user_modules(
    schema_file: str,
    cleaners_file: Optional[str] = None,
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

    if cleaners_file is not None:
        cleaners_module = import_module_from_path(cleaners_file)
        cleaners = get_cleaners(cleaners_module)
    else:
        cleaners = []

    return schema, cleaners


def get_args(config_file: str) -> DataConfig:
    """Gets arguments from config.yml."""
    with open(config_file, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return DataConfig(**config)


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
) -> pd.DataFrame:
    """Reads a file."""
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return read_excel_file(filename, sheet_name=sheet_name, skiprows=skip_rows)
    if filename.endswith(".csv"):
        return pd.read_csv(filename, skiprows=skip_rows)

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
    input_file: str,
    input_file_config: InputFileConfig,
) -> pd.DataFrame:
    """Reads an input file, or gets the output of a preprocessor function."""
    if input_file_config.preprocessor is not None:
        df = call_preprocess_from_file(
            input_file_config.preprocessor.path,
            input_file_config.preprocessor.kwargs,
        )
    else:
        df = read_file(
            input_file,
            input_file_config.sheet_name,
            input_file_config.skip_rows,
        )
    return df
