"""Doctor command - System health checks."""

import logging

import typer

from voicepad.cli.checks.config import check_config

logger = logging.getLogger(__name__)

doctor_app = typer.Typer(help="System health and configuration checks")


@doctor_app.command()
def config() -> None:
    check_config()


def run_all_checks() -> None:
    typer.echo("Running System Checks...")

    try:
        check_config()
        typer.echo()
    except typer.Exit:
        raise

    typer.secho("✓ All checks completed", fg=typer.colors.GREEN)


@doctor_app.callback(invoke_without_command=True)
def doctor_main(ctx: typer.Context) -> None:
    # If no subcommand is specified, run all checks
    if ctx.invoked_subcommand is None:
        run_all_checks()
