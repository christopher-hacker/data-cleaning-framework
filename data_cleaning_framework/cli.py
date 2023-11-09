"""Command line interface for data cleaning framework"""
import click


@click.group()
def cli():
    """Command line interface for data cleaning framework"""
    pass  # pylint: disable=unnecessary-pass


@click.command()
@click.option(
    "--test-run",
    is_flag=True,
    help="Run in test mode, cleaning only the last file in the config file",
)
@click.option(
    "--threads",
    type=int,
    default=1,
    help="Number of threads to use for cleaning. Specifying more than 1 will "
    "run multiple files in parallel. Defaults to 1.",
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
def clean_data(
    test_run,
    threads,
    scenario,
):
    """Run a data cleaning task with a given config file"""
    click.echo(
        f"clean_data with test_run: {test_run}, threads: {threads}, scenario: {scenario}"
    )


cli.add_command(clean_data)
