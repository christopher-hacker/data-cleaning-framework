"""Contains utility functions for handling cleaners."""

import inspect
import sys
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


def get_cleaners(scenario: Optional[str] = None) -> List[tuple]:
    """Returns a list of cleaner functions and their arguments sorted by their order."""
    default_scenarios = [None, "default"]
    all_cleaners = []
    # get all the functions in this module that have been decorated with the cleaner decorator
    for _, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and hasattr(obj, "is_cleaner"):
            all_cleaners.append(obj)

    use_cleaners = list(
        filter(lambda x: x.cleaner_args.scenario in default_scenarios, all_cleaners)
    )

    if scenario not in default_scenarios:
        scenario_cleaners = list(
            filter(
                lambda x, scenario=scenario: x.cleaner_args.scenario == scenario,
                all_cleaners,
            )
        )
        # make sure there's at least one cleaner in the scenario
        if len(list(scenario_cleaners)) == 0:
            raise ValueError(f"No cleaners found with scenario '{scenario}'")
        # otherwise, add the cleaners to the list
        use_cleaners.extend(scenario_cleaners)

    # sort the functions by their order
    use_cleaners = sorted(use_cleaners, key=lambda x: getattr(x, "order", float("inf")))

    # Return the sorted functions along with their cleaner_args
    return [(func, func.cleaner_args) for func in use_cleaners]
