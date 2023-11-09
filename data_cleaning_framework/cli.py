"""Command line interface for data cleaning framework"""
import click


@click.group()
def cli():
    """Command line interface for data cleaning framework"""
    pass  # pylint: disable=unnecessary-pass


@click.command()
@click.argument("config_file")
def run_task(config_file):
    """Run a data cleaning task with a given config file"""
    click.echo(f"run_task with config_file: {config_file}")


cli.add_command(run_task)
