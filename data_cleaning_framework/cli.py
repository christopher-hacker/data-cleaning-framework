"""Command line interface for data cleaning framework"""
import click
from . import clean_data as clean_data_module


@click.group()
def cli():
    """Command line interface for data cleaning framework"""
    pass  # pylint: disable=unnecessary-pass


@click.command()
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
