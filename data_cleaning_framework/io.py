"""Reads and writes data."""

import builtins
import importlib.util
from typing import Optional, Union
import pandas as pd
import yaml
from data_cleaning_framework.models import DataConfig


def insert_into_namespace(module_path, module_name, *object_names):
    """Imports a module from a file and inserts it into the global namespace."""
    spec = importlib.util.spec_from_file_location("module.name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if object_names:
        for object_name in object_names:
            globals()[object_name] = getattr(module, object_name)
            builtins.__dict__[object_name] = getattr(module, object_name)
    else:
        globals()[module_name] = module
        builtins.__dict__[module_name] = module


def load_user_modules(
    schema_file: Optional[str] = None,
    cleaners_file: Optional[str] = None,
) -> None:
    """Loads user-defined modules."""
    if schema_file is not None:
        insert_into_namespace(schema_file, "schema", "Schema")
    else:
        from schema import Schema  # type: ignore # pylint: disable=import-error,unused-import,import-outside-toplevel,line-too-long

    if cleaners_file is not None:
        insert_into_namespace(cleaners_file, "cleaners")
    else:
        import cleaners  # type: ignore # pylint: disable=import-error,unused-import,import-outside-toplevel,line-too-long


def get_args(config_file: str = "config.yml") -> DataConfig:
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
        return read_excel_file(filename, sheet_name=sheet_name, skip_rows=skip_rows)
    if filename.endswith(".csv"):
        return pd.read_csv(filename, skip_rows=skip_rows)

    raise ValueError(f"Unsupported file type: {filename}")
