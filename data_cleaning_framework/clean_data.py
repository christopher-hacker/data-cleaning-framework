"""Cleans data based on a single config file."""

from functools import partial, wraps
import importlib
import multiprocessing
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
import pandera as pa
from pydantic import validate_call
from tqdm import tqdm
import yaml
from .cleaner_utils import get_cleaners
from .log import get_logger
from .models import InputFileConfig, DataConfig

logger = get_logger()


def log_processor(func: Callable) -> Callable:
    """Decorator to log the length of the DataFrame passed to a processor function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function."""
        try:
            result = func(*args, **kwargs)
            assert isinstance(result, pd.DataFrame), (
                f"Function {func.__name__} must return a DataFrame, "
                f"but got {type(result)}"
            )
            logger.info(
                "Function %s resulted in DataFrame with shape [%d, %d]. "
                "Args: %s, kwargs: %s",
                func.__name__,
                result.shape[0],
                result.shape[1],
                ", ".join(
                    [
                        str(arg) if not isinstance(arg, pd.DataFrame) else "<DataFrame>"
                        for arg in args
                    ]
                ),
                ", ".join(
                    [
                        (
                            f"{key}={value}"
                            if not isinstance(value, pd.DataFrame)
                            else f"{key}=<DataFrame>"
                        )
                        for key, value in kwargs.items()
                    ]
                ),
            )
            return result
        except Exception as exc:
            logger.error(
                "Error while calling function %s: %s", func.__name__, repr(exc)
            )
            raise exc

    return wrapper


def load_user_modules(
    schema_file: Optional[str] = None,
    cleaners_file: Optional[str] = None,
) -> None:
    """Loads user-defined modules."""
    if schema_file is not None:
        spec = importlib.util.spec_from_file_location("schema", schema_file)
        schema_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema_module)
        globals()["Schema"] = schema_module.Schema
    else:
        from schema import Schema  # type: ignore # pylint: disable=import-error,unused-import,import-outside-toplevel,line-too-long

    if cleaners_file is not None:
        spec = importlib.util.spec_from_file_location("cleaners", cleaners_file)
        cleaners_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cleaners_module)
        globals()["get_cleaners"] = cleaners_module.get_cleaners
    else:
        import cleaners  # type: ignore # pylint: disable=import-error,unused-import,import-outside-toplevel,line-too-long


class CleaningFailedError(Exception):
    """Exception raised when cleaning fails?"""

    pass  # pylint: disable=unnecessary-pass


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

        raise exc

    return df


def read_csv_file(
    filename: str, skip_rows: Optional[int] = None, **kwargs
) -> pd.DataFrame:
    """Reads a CSV file."""
    try:
        df = pd.read_csv(filename, skiprows=skip_rows, **kwargs)
    except ValueError as exc:
        raise ValueError(
            f"Error while reading {filename}. Please check the config file."
        ) from exc

    return df


def read_file(
    filename: str,
    sheet_name: Union[str, int] = 0,
    skip_rows: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Reads a file."""
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return read_excel_file(filename, sheet_name, skip_rows, **kwargs)
    if filename.endswith(".csv"):
        return read_csv_file(filename, skip_rows, **kwargs)

    raise ValueError(f"Unsupported file type: {filename}")


@log_processor
@validate_call
def assign_columns(
    df: pd.DataFrame,
    *columns: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Assigns values to columns in a DataFrame."""
    for column_dict in columns:
        df = df.assign(**column_dict)
    return df


@log_processor
@validate_call
def rename_columns(
    df: pd.DataFrame, columns: Dict[Union[int, str], str]
) -> pd.DataFrame:
    """Renames columns in a DataFrame."""
    # check that none of the provided columns are missing
    for old_name_or_index in columns.keys():
        if isinstance(old_name_or_index, int):
            if old_name_or_index >= len(df.columns):
                raise ValueError(
                    f"Column index {old_name_or_index} is out of range. "
                    f"DataFrame has {len(df.columns)} columns."
                )
        else:
            if old_name_or_index not in df.columns:
                raise ValueError(
                    f"Column name {old_name_or_index} not found in DataFrame. "
                    f"Available columns: {df.columns}"
                )

    for old_name_or_index, new_name in columns.items():
        # check that the new name is not already in the DataFrame
        # and that it is in the schema
        if new_name in df.columns:
            raise ValueError(
                f"Column name {new_name} is already in DataFrame. "
                "Please check the config file."
            )
        if new_name not in Schema.to_schema().columns:
            raise ValueError(
                f"Column name {new_name} not found in schema. "
                "Please check the config file."
            )

        if isinstance(old_name_or_index, int):
            df = df.rename(columns={df.columns[old_name_or_index]: new_name})
        else:
            df = df.rename(columns={old_name_or_index: new_name})

    # check that there are no duplicate column names
    if len(df.columns) != len(set(df.columns)):
        raise ValueError(
            "Renaming columns resulted in duplicate column names. "
            "Please check the config file."
        )
    return df


@log_processor
@validate_call
def drop_rows(df: pd.DataFrame, rows: Optional[List[int]] = None) -> pd.DataFrame:
    """Drops rows from a DataFrame."""
    if rows is not None:
        df = df.drop(rows)
    return df


@log_processor
@validate_call
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes columns in a DataFrame match the schema.
    """

    colnames = list(Schema.to_schema().columns.keys())
    missing_cols = [col for col in colnames if col not in df.columns]

    if missing_cols:
        missing_data = pd.DataFrame(np.nan, index=df.index, columns=missing_cols)
        df = pd.concat([df, missing_data], axis=1)[colnames]

    return df


@log_processor
@validate_call
def replace_values(
    df: pd.DataFrame,
    value_mapping: Optional[Dict[str, Dict[Union[int, str], Union[int, str]]]] = None,
):
    """Replaces values in a DataFrame."""
    if value_mapping is not None:
        for column_name, replacements in value_mapping.items():
            df[column_name] = df[column_name].replace(replacements)
    return df


@log_processor
@validate_call
def apply_query(df: pd.DataFrame, query: Optional[str] = None) -> pd.DataFrame:
    """Applies a query to a DataFrame."""
    if query is None:
        return df

    try:
        return df.query(query)
    except pd.errors.UndefinedVariableError as exc:
        raise ValueError(
            f"Error while applying query '{query}' to DataFrame. "
            "Please check the query and the column names in the config file. \n\n"
            f"Available columns: {df.columns}"
        ) from exc


def call_preprocess_from_file(
    relative_path: str,
    kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Calls the function 'preprocess' from a provided python file."""
    spec = importlib.util.spec_from_file_location("module.name", relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "preprocess"):
        if kwargs is None:
            return module.preprocess()

        return module.preprocess(**kwargs)

    raise ValueError(
        f"The function 'preprocess' was not found in the given file '{relative_path}'"
    )


@log_processor
@validate_call
def get_input_file(
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


@log_processor
@validate_call
def apply_cleaners(df: pd.DataFrame, scenario: Optional[str] = None) -> pd.DataFrame:
    """Applies all cleaners to a DataFrame."""
    for func, args in get_cleaners(scenario=scenario):
        cleaner_called = False
        if args.dataframe_wise is True:
            df = func(df)
            logger.info(
                "Dataframe-wise cleaner %s resulted in DataFrame with shape [%d, %d].",
                func.func_name,
                df.shape[0],
                df.shape[1],
            )
            assert (
                len(df) > 0
            ), f"Dataframe is empty after applying cleaner '{func.func_name}'"
            cleaner_called = True

        elif args.columns is not None:
            for column_name in args.columns:
                # skip if not in dataframe
                if column_name not in df.columns:
                    continue

                df[column_name] = df[column_name].apply(func)
                logger.info(
                    "Column-wise cleaner %s applied to column %s.",
                    func.func_name,
                    column_name,
                )
                cleaner_called = True

        elif args.dtypes is not None:
            # apply the function to all columns with the specified dtypes
            # in the schema
            schema_columns = Schema.to_schema().columns
            for column_name, column in schema_columns.items():
                # skip if not in dataframe, which is fine if
                # this is being called after the columns are renamed
                if column_name not in df.columns:
                    continue

                for dtype in args.dtypes:
                    if column.dtype.type == dtype:
                        df[column_name] = df[column_name].apply(func)
                        logger.info(
                            "Column-wise cleaner %s applied to column %s with dtype %s.",
                            func.func_name,
                            column_name,
                            dtype,
                        )
                        cleaner_called = True

        if not cleaner_called:
            raise ValueError(
                f"Cleaner {func.func_name} was not applied to any columns. "
                "Please check the function decorator in src/cleaners.py."
            )

    return df


@pa.check_types
def process_single_file(
    input_file_config: InputFileConfig,
    args: DataConfig,
    scenario: Optional[str] = None,
) -> pd.DataFrame:
    """Processes a single file."""
    return (
        get_input_file(input_file_config.input_file, input_file_config)
        # drop any rows specified in the config file
        .pipe(drop_rows, input_file_config.drop_rows)
        # rename the column from the mapping provided
        .pipe(rename_columns, input_file_config.rename_columns)
        # optionally query if provided
        .pipe(
            apply_query,
            input_file_config.query,
        )
        # assign columns provided in the config file
        .pipe(
            assign_columns,
            *[
                d
                for d in [args.assign_columns, input_file_config.assign_columns]
                if d is not None
            ],
        )
        # replace any values if provided
        .pipe(replace_values, input_file_config.replace_values)
        # add columns if they are missing
        # and reorder columns to match schema
        .pipe(standardize_columns)
        # apply cleaners
        .pipe(apply_cleaners, scenario=scenario)
        # apply schema validation
        .pipe(Schema.to_schema().validate)
    )


def process_and_write_file(
    input_file_config: InputFileConfig,
    yaml_args: DataConfig,
    scenario: Optional[str],
    lock: Callable,
):
    """Processes and writes a file, used to parallelize the process."""
    try:
        processed_data = process_single_file(input_file_config, yaml_args, scenario)
        with lock:
            processed_data.to_csv(
                yaml_args.output_file, index=False, mode="a", header=False
            )
    except Exception as exc:
        error_message = "Error while cleaning file. "
        if input_file_config.input_file is not None:
            error_message += f"Please check the file {input_file_config.input_file}."
        if input_file_config.preprocessor is not None:
            error_message += (
                f"Please check the preprocess function in "
                f"{input_file_config.preprocessor.path}. "
                f"Args: {input_file_config.preprocessor.kwargs}."
            )
        raise CleaningFailedError(error_message) from exc


def main(
    threads: int,
    test_run: bool,
    scenario: str,
    config_file: str = "config.yml",
    schema_file: Optional[str] = None,
    cleaners_file: Optional[str] = None,
) -> None:
    """Cleans data from a single source."""
    logger = get_logger(scenario)  # pylint: disable=redefined-outer-name
    yaml_args = get_args(config_file)

    # schema and cleaner files can be provided as arguments, or they can be specified
    # in the config file. If they're both provided, fail loudly
    if schema_file is not None and yaml_args.schema_file is not None:
        raise ValueError(
            "Schema file provided as argument and in config file. "
            "Please provide only one."
        )
    if cleaners_file is not None and yaml_args.cleaners_file is not None:
        raise ValueError(
            "Cleaners file provided as argument and in config file. "
            "Please provide only one."
        )

    # if they're not provided as arguments, use the ones from the config file
    # if they're null in both places, the default behavior will still be used
    if schema_file is None:
        schema_file = yaml_args.schema_file

    if cleaners_file is None:
        cleaners_file = yaml_args.cleaners_file

    load_user_modules(schema_file=schema_file, cleaners_file=cleaners_file)

    if test_run:
        yaml_args.input_files = yaml_args.input_files[-1:]

    if scenario is not None:
        logger.info("Running scenario '%s'.", scenario)
        # append the version name to the output file name
        yaml_args.output_file = yaml_args.output_file.replace(
            ".csv", f"-{scenario}.csv"
        )

    # write just the header to the output file
    pd.DataFrame(columns=Schema.to_schema().columns).to_csv(
        yaml_args.output_file, index=False
    )

    pbar = tqdm(
        desc=f"{len(yaml_args.input_files)} files -> {yaml_args.output_file}",
        total=len(yaml_args.input_files),
    )
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    # Create a partial function with some arguments pre-filled
    partial_process = partial(
        process_and_write_file, yaml_args=yaml_args, scenario=scenario, lock=lock
    )
    # Create a pool of worker threads
    with multiprocessing.Pool(processes=threads) as pool:
        # Use the pool's map method to process the files in parallel
        for _ in pool.imap_unordered(partial_process, yaml_args.input_files):
            pbar.update()
