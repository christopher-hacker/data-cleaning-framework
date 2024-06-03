"""Contains utility functions for handling cleaners."""

from typing import List, Optional, Union
import pydantic


class CleanerArgs(pydantic.BaseModel):
    """Arguments for a cleaner."""

    columns: Optional[List[str]] = None
    dtypes: Optional[List[type]] = None
    dataframe_wise: bool = False
    order: int = 0
    scenario: Optional[Union[str, List[str]]] = None


@pydantic.validate_call
def cleaner(
    columns: Optional[List[str]] = None,
    dtypes: Optional[List[type]] = None,
    dataframe_wise: bool = False,
    order: int = 0,
    scenario: Optional[Union[str, List[str]]] = None,
):
    """Decorator that declares a function as a cleaner."""

    # Validate the arguments
    cleaner_args = CleanerArgs(
        columns=columns,
        dtypes=dtypes,
        dataframe_wise=dataframe_wise,
        order=order,
        scenario=scenario,
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
            return func(*args, **kwargs)

        # Tag the wrapped function with the arguments passed to the decorator
        # This allows us to inspect the function later to see what
        # arguments were passed to the decorator
        wrapper.is_cleaner = True
        wrapper.cleaner_args = cleaner_args
        wrapper.func_name = func.__name__  # pylint: disable=protected-access
        wrapper.order = order
        wrapper.scenario = scenario
        return wrapper

    return decorator
