"""Cleans data based on a single config file."""

from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from loguru import logger
import numpy as np
import pandas as pd
import pandera as pa
from pydantic import validate_call
from tqdm import tqdm
from .io import load_user_modules, get_args, load_data, write_data
from .models import InputFileConfig, DataConfig, Any
from .constants import APPENDABLE_OUTPUT_TYPES

logger.remove()


def get_info_as_string(df: pd.DataFrame) -> None:
    """Gets the shape of a DataFrame as a string for logging."""
    buffer = StringIO()
    df.info(buf=buffer, verbose=True, show_counts=True)
    return buffer.getvalue()


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
            function_name = func.__name__
            args_str = ", ".join(
                [
                    str(arg) if not isinstance(arg, pd.DataFrame) else "<DataFrame>"
                    for arg in args
                ]
            )
            kwargs_str = ", ".join(
                [
                    (
                        f"{key}={value}"
                        if not isinstance(value, pd.DataFrame)
                        else f"{key}=<DataFrame>"
                    )
                    for key, value in kwargs.items()
                ]
            )
            logger.info(
                f"Called function {function_name} with args: {args_str}, kwargs: {kwargs_str}"
                f"\nDataframe info: \n{get_info_as_string(result)}"
            )
            return result
        except Exception as exc:
            logger.error(f"Error while calling function {func.__name__}: {repr(exc)}")
            raise exc

    return wrapper


@log_processor
@validate_call
def assign_constant_columns(
    df: Any,
    *columns: Dict[str, Any],
) -> pd.DataFrame:
    """
    Assigns constant values to columns in a DataFrame.Useful for things like
    adding a source column to a dataframe that will be merged with other dataframes.
    """
    for column_dict in columns:
        df = df.assign(**column_dict)
    return df


@log_processor
@validate_call
def rename_columns(
    df: Any,
    columns: Optional[Dict[Union[int, str], str]] = None,
) -> pd.DataFrame:
    """Renames columns in a DataFrame."""
    # Allow the function to be called without renaming columns
    if columns is None:
        return df

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

        if isinstance(old_name_or_index, int):
            df = df.rename(columns={df.columns[old_name_or_index]: new_name})
        else:
            df = df.rename(columns={old_name_or_index: new_name})

    return df


@log_processor
@validate_call
def drop_rows(df: Any, rows: Optional[List[int]] = None) -> pd.DataFrame:
    """Drops rows from a DataFrame."""
    if rows is not None:
        df = df.drop(rows).reset_index(drop=True)
    return df


@log_processor
@validate_call
def drop_columns(
    df: Any, columns: Optional[List[Union[str, int]]] = None
) -> pd.DataFrame:
    """Drops columns from a DataFrame."""
    if columns is not None:
        df = df.drop(columns, axis=1)
    return df


@log_processor
@validate_call
def add_missing_columns(df: Any, valid_columns: List[str]) -> pd.DataFrame:
    """
    Makes columns in a DataFrame match the schema.
    """

    missing_cols = [col for col in valid_columns if col not in df.columns]

    if missing_cols:
        missing_data = pd.DataFrame(np.nan, index=df.index, columns=missing_cols)
        df = pd.concat([df, missing_data], axis=1)[valid_columns].reset_index(drop=True)

    return df


@log_processor
@validate_call
def drop_extra_columns(df: Any, valid_columns: List[str], ignore=True) -> pd.DataFrame:
    """
    Drops columns from a DataFrame that are not in the schema.
    """
    if ignore:
        return df

    extra_cols = [col for col in df.columns if col not in valid_columns]

    if extra_cols:
        df = df.drop(columns=extra_cols)

    return df


@log_processor
@validate_call
def replace_values(
    df: Any,
    value_mapping: Optional[Dict[str, Dict[Any, Any]]],
):
    """Replaces values in a DataFrame."""
    if value_mapping is not None:
        for column_name, replacements in value_mapping.items():
            df[column_name] = df[column_name].replace(replacements)
    return df


@log_processor
@validate_call
def apply_query(df: Any, query: Optional[str]) -> pd.DataFrame:
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


@log_processor
@validate_call
def parse_date_columns(
    df: Any,
    date_columns: Optional[Dict[str, str]] = None,
    errors: Optional[str] = "raise",
) -> pd.DataFrame:
    """Parses date columns in a DataFrame."""
    if date_columns is None:
        return df

    for column_name, date_format in date_columns.items():
        try:
            df[column_name] = pd.to_datetime(
                df[column_name], format=date_format, errors=errors
            )
        except KeyError as exc:
            raise ValueError(
                f"Error while parsing date column '{column_name}'. "
                "Please check the column name in the config file."
            ) from exc
    return df


@log_processor
@validate_call
def apply_cleaners(
    df: Any,
    cleaners: List[Tuple[Callable, Any]],
    schema_columns: Dict[str, pa.Column],
) -> pd.DataFrame:
    """Applies all cleaners to a DataFrame."""

    def apply_dataframe_wise_cleaner(df, func):
        df = func(df)
        logger.info(
            f"Dataframe-wise cleaner {func.func_name} resulted in DataFrame "
            f"with shape [{df.shape[0]}, {df.shape[1]}]."
        )
        assert (
            len(df) > 0
        ), f"Dataframe is empty after applying cleaner '{func.func_name}'"
        return df

    def apply_column_wise_cleaner(df, func, columns):
        for column_name in columns:
            if column_name in df.columns:
                df[column_name] = df[column_name].apply(func)
                logger.info(
                    f"Column-wise cleaner {func.func_name} applied to column {column_name}."
                )
            else:
                logger.warning(
                    f"Cleaner {func.func_name} was not applied to any columns "
                    "because none of the columns matched the specified dtypes."
                )
        return df

    def apply_dtype_wise_cleaner(df, func, dtypes, schema_columns):
        if any(column.dtype.type in dtypes for column in schema_columns.values()):
            for column_name, column in schema_columns.items():
                for dtype in dtypes:
                    if column.dtype.type == dtype:
                        df[column_name] = df[column_name].apply(func)
                        logger.info(
                            f"Column-wise cleaner {func.func_name} applied to "
                            f"column {column_name} with dtype {dtype}."
                        )
        else:
            logger.warning(
                f"Cleaner {func.func_name} was not applied to any columns "
                "because none of the columns matched the specified dtypes."
            )
        return df

    for func, args in cleaners:
        if args.dataframe_wise is True:
            df = apply_dataframe_wise_cleaner(df, func)
        elif args.columns is not None:
            df = apply_column_wise_cleaner(df, func, args.columns)
        elif args.dtypes is not None:
            df = apply_dtype_wise_cleaner(df, func, args.dtypes, schema_columns)

        logger.info(
            f"Dataframe info after applying cleaner {func.func_name}: \n"
            + get_info_as_string(df)
        )

    return df


@pa.check_types
def process_single_file(
    input_file_config: InputFileConfig,
    args: DataConfig,
    valid_columns: List[str],
    schema: pa.SchemaModel,
    cleaners: Optional[List[Callable]] = None,
) -> pd.DataFrame:
    """Processes a single file."""
    logger.info(f"\n#####\n##### Cleaning file: {input_file_config.input_file}\n#####")
    return (
        load_data(input_file_config, logger=logger)
        # drop any rows specified in the config file
        .pipe(drop_rows, input_file_config.drop_rows)
        # drop any columns specified in the config file
        .pipe(drop_columns, input_file_config.drop_columns)
        # rename the column from the mapping provided
        .pipe(
            rename_columns,
            columns=input_file_config.rename_columns,
        )
        # parse datetime columns
        .pipe(
            parse_date_columns,
            date_columns=input_file_config.date_columns,
            errors=input_file_config.date_errors,
        )
        # replace any values if provided
        .pipe(replace_values, input_file_config.replace_values)
        # optionally query if provided
        .pipe(
            apply_query,
            input_file_config.query,
        )
        # assign columns provided in the config file
        .pipe(
            assign_constant_columns,
            *[
                d
                for d in [
                    args.assign_constant_columns,
                    input_file_config.assign_constant_columns,
                ]
                if d is not None
            ],
        )
        # apply cleaners
        .pipe(
            apply_cleaners, cleaners=cleaners, schema_columns=schema.to_schema().columns
        )
        # drop columns not in the schema
        .pipe(
            drop_extra_columns,
            valid_columns=valid_columns,
            ignore=not input_file_config.drop_extra_columns,
        )
        # add columns if they are missing
        # and reorder columns to match schema
        .pipe(add_missing_columns, valid_columns=valid_columns)
        # apply schema validation
        .pipe(log_processor(schema.to_schema().validate))
    )


def process_config(yaml_args: DataConfig) -> None:
    """Runs processing for one dataset"""
    schema, cleaners = load_user_modules(
        schema_file=yaml_args.schema_file, cleaners_files=yaml_args.cleaners_files
    )

    # get the list of valid columns from schema, used throughout the process
    valid_columns = schema.to_schema().columns
    valid_columns = list(valid_columns.keys())

    # write just the header to the output file
    pd.DataFrame(columns=valid_columns).to_csv(yaml_args.output_file, index=False)

    pbar = tqdm(
        desc=f"{len(yaml_args.input_files)} files -> {yaml_args.output_file}",
        total=len(yaml_args.input_files),
    )

    output_file_extension = Path(yaml_args.output_file).suffix
    chunks = []
    for file_index, input_file_config in enumerate(
        yaml_args.input_files
    ):  # pylint: disable=not-an-iterable
        if input_file_config.cleaners_files:
            _, file_cleaners = load_user_modules(
                schema_file=yaml_args.schema_file,
                cleaners_files=input_file_config.cleaners_files,
            )
        else:
            file_cleaners = []

        result = process_single_file(
            input_file_config=input_file_config,
            args=yaml_args,
            valid_columns=valid_columns,
            schema=schema,
            cleaners=cleaners + file_cleaners,
        )
        # if the file is a csv or other type that can be written to without reading the whole
        # file into memory, write it directly to the output file
        if output_file_extension in APPENDABLE_OUTPUT_TYPES:
            write_data(
                result,
                yaml_args.output_file,
                logger=logger,
                append=file_index > 0,
            )
        else:
            chunks.append(result)
        pbar.update()

    if chunks:
        # if being loaded into memory, check against the schema after concatenating
        # to make sure that the schema is still valid
        df = pd.concat(chunks)
        schema.to_schema().validate(df)
        write_data(
            df,
            yaml_args.output_file,
            logger=logger,
            append=False,
        )


def main(config_file: str) -> None:
    """Cleans data from a single source."""

    yaml_args = get_args(config_file)

    # log to clean_data.log in the same directory as the config file

    config_dir_path = Path(config_file).resolve().parent
    logger.remove()
    logger.add(config_dir_path / "clean_data.log")

    logger.info(
        "\n#####"
        "\n#####"
        "\n##### Begin cleaning run"
        f"\n##### Using config file: {config_file}"
        "\n#####"
        "\n#####\n"
    )

    if isinstance(yaml_args, list):
        for args in yaml_args:
            process_config(args)
    else:
        process_config(yaml_args)

    logger.info("\n#####\n#####\n##### End cleaning run\n#####\n#####\n")
