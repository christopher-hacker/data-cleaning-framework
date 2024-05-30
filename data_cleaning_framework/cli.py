"""Command line interface for data cleaning framework"""

from typing import Dict, Any
import click
import yaml
from . import clean_data as clean_data_module
from .models import DataConfig


def reorder_keys(data: Dict) -> Dict:
    """Reorders the keys in a dictionary."""
    # Define the desired order of keys
    key_order = ["title", "type", "description", "items"]
    # Create a new dictionary to store the reordered keys
    reordered = {}
    # Insert the keys in the desired order
    for key in key_order:
        if key in data:
            reordered[key] = data.pop(key)
    # Insert any remaining keys
    for key, value in data.items():
        reordered[key] = value
    return reordered


def get_defs_props(val: Dict[Any, Any], json_data: Dict[Any, Any]) -> Dict[Any, Any]:
    """Get the properties from the $ref key in the definitions section of the JSON schema."""
    if "items" in val and "$ref" in val["items"]:
        # Get the reference key
        ref_key = val["items"]["$ref"].split("/")[-1]
        # Get the properties from the reference key
        val_props = json_data["$defs"][ref_key]["properties"]
    else:
        val_props = val  # return original value if no $ref is found

    # Handle nested $ref within anyOf key
    for key, prop_val in val_props.items():
        if "anyOf" in prop_val:
            updated_anyof = []
            for option in prop_val["anyOf"]:
                if "$ref" in option:
                    ref_key = option["$ref"].split("/")[-1]
                    ref_props = json_data["$defs"][ref_key]["properties"]
                    updated_anyof.append(ref_props)
                else:
                    updated_anyof.append(option)
            prop_val["anyOf"] = updated_anyof
        # Reorder the keys
        val_props[key] = reorder_keys(prop_val)

    return val_props


def get_help_message(json_data: Dict[Any, Any]) -> str:
    """Gets the config help message."""
    # Make a copy of the original properties dictionary
    original_properties = json_data["properties"].copy()

    # Create a new dictionary to store the updated properties
    updated_properties = {}

    # Iterate through the original properties dictionary
    for key, val in original_properties.items():
        # If the "items" key is present, update its value
        if "items" in val:
            val["items"] = get_defs_props(val, json_data)

        # Reorder the keys
        val = reorder_keys(val)

        # Insert the updated key-value pair into the updated properties dictionary
        updated_properties[key] = val

    # Update the original JSON data with the updated properties dictionary
    json_data["properties"] = updated_properties

    return yaml.dump(json_data["properties"], sort_keys=False)


class CustomHelpCommand(click.Command):
    """Custom command that prints the custom help message."""

    def get_help(self, ctx):
        default_help_message = super().get_help(ctx)
        custom_help_message = (
            "Must be run in a project containing a config file named 'config.yml'. "
            + "\n\nConfig file options:\n\n"
            + get_help_message(DataConfig.model_json_schema())
        )
        return f"{default_help_message}\n{custom_help_message}"


@click.group()
def cli():
    """Command line interface for data cleaning framework"""
    pass  # pylint: disable=unnecessary-pass


@click.command(cls=CustomHelpCommand)
@click.option(
    "--threads",
    type=int,
    default=1,
    help="Number of threads to use for cleaning. Specifying more than 1 will "
    "run multiple files in parallel. Defaults to 1.",
)
@click.option(
    "--test-run",
    is_flag=True,
    help="Run in test mode, cleaning only the last file in the config file",
)
@click.option(
    "--scenario",
    type=str,
    default="default",
    help="The name of a scenario to run extra cleaners from. Must be a name specified in the "
    "cleaner decorator of at least one cleaner function. All default cleaners will be run "
    "regardless of this option. If not provided, only default cleaners will be run. "
    "You can also provide the string 'default' to run with the same cleaners as the default.",
)
@click.option(
    "--config-file",
    type=str,
    default="config.yml",
    help="The path to the config file to use. Defaults to config.yml.",
)
# getting definitions from the DataConfig class so that the help text is consistent
@click.option(
    "--schema_file",
    type=str,
    default=(
        clean_data_module.DataConfig.model_json_schema()["properties"]["schema_file"][
            "default"
        ]
    ),
    help=(
        clean_data_module.DataConfig.model_json_schema()["properties"]["schema_file"][
            "description"
        ]
    ),
)
@click.option(
    "--cleaners_file",
    type=str,
    default=(
        clean_data_module.DataConfig.model_json_schema()["properties"]["cleaners_file"][
            "default"
        ]
    ),
    help=(
        clean_data_module.DataConfig.model_json_schema()["properties"]["cleaners_file"][
            "description"
        ]
    ),
)
def clean_data(
    test_run,
    threads,
    scenario,
    config_file,
    schema_file,
    cleaners_file,
):
    """Run a data cleaning task with a given config file"""
    clean_data_module.main(
        test_run=test_run,
        threads=threads,
        scenario=scenario,
        config_file=config_file,
        schema_file=schema_file,
        cleaners_file=cleaners_file,
    )


cli.add_command(clean_data)
