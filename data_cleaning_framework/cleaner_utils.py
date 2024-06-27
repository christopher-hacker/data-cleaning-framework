"""Contains utility functions for handling cleaners."""

from typing import List, Optional
import pandas as pd
import pydantic
from .models import CleanerArgs


@pydantic.validate_call
def cleaner(
    columns: Optional[List[str]] = None,
    dtypes: Optional[List[type]] = None,
    dataframe_wise: bool = False,
    order: int = 0,
    return_nulls: bool = False,
):
    """Decorator that declares a function as a cleaner."""

    # Validate the arguments
    cleaner_args = CleanerArgs(
        columns=columns,
        dtypes=dtypes,
        dataframe_wise=dataframe_wise,
        order=order,
        return_nulls=return_nulls,
    )

    methods = [columns, dtypes, dataframe_wise]
    if not any(methods):
        raise ValueError(
            "At least one of `columns`, `dtypes`, or `dataframe_wise` must be specified."
        )
    provided_methods = list(
        filter(
            lambda x: x is True,
            [
                columns is not None,
                dtypes is not None,
                dataframe_wise is True,
            ],
        )
    )
    if len(provided_methods) > 1:
        raise ValueError(
            "Only one of `columns`, `dtypes`, or `dataframe_wise` can be specified."
        )

    def decorator(func):
        def wrapper(*args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0:
                # if there is only one argument and that argument is null and return_nulls
                # is true, then return the value without calling the cleaner function
                if return_nulls and pd.isna(args[0]):
                    return None
            return func(*args, **kwargs)

        # Tag the wrapped function with the arguments passed to the decorator
        # This allows us to inspect the function later to see what
        # arguments were passed to the decorator
        wrapper.is_cleaner = True
        wrapper.cleaner_args = cleaner_args
        wrapper.func_name = func.__name__  # pylint: disable=protected-access
        wrapper.order = order
        wrapper.return_nulls = return_nulls
        return wrapper

    return decorator
