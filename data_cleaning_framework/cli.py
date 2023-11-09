import click


@click.group()
def cli():
    pass


@click.command()
@click.argument("config_file")
def run_task(config_file):
    click.echo(f"run_task with config_file: {config_file}")


cli.add_command(run_task)
