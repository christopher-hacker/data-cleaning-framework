import click

@click.group()
def cli():
    pass

@click.command()
@click.argument('argument')
def run(argument):
    click.echo(f'Running with argument: {argument}')

cli.add_command(run)
