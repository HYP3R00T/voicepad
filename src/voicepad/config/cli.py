"""CLI commands for configuration management."""

import logging

import typer

from voicepad.config.settings import get_config

logger = logging.getLogger(__name__)

config_app = typer.Typer(help="Configuration management commands")


@config_app.command()
def show() -> None:
    """Display current configuration."""
    config = get_config()

    typer.echo("Current configuration:")
    typer.echo(f"  Recordings path: {config.recordings_path}")
    typer.echo(f"  Markdown path: {config.markdown_path}")


@config_app.command()
def verify() -> None:
    """Verify configuration and directories."""
    config = get_config()

    typer.echo("Verifying configuration...")

    # Check recordings path
    if config.recordings_path.exists():
        typer.echo(f"✓ Recordings path exists: {config.recordings_path}")
    else:
        typer.echo(f"✗ Recordings path not found: {config.recordings_path}")
        try:
            config.recordings_path.mkdir(parents=True, exist_ok=True)
            typer.echo(f"✓ Created recordings path: {config.recordings_path}")
        except Exception as e:
            typer.echo(f"✗ Failed to create recordings path: {e}", err=True)

    # Check markdown path
    if config.markdown_path.exists():
        typer.echo(f"✓ Markdown path exists: {config.markdown_path}")
    else:
        typer.echo(f"✗ Markdown path not found: {config.markdown_path}")
        try:
            config.markdown_path.mkdir(parents=True, exist_ok=True)
            typer.echo(f"✓ Created markdown path: {config.markdown_path}")
        except Exception as e:
            typer.echo(f"✗ Failed to create markdown path: {e}", err=True)

    typer.echo("✓ Configuration verified!")
