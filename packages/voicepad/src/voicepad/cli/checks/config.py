"""Configuration system checker."""

import logging
from pathlib import Path

import typer
from voicepad_core import get_config_with_metadata

logger = logging.getLogger(__name__)


def check_config() -> None:
    """Check configuration file status and location.

    Reports:
    - Whether external config file is found
    - Path to the config file being used
    - Precedence order being followed
    """
    try:
        config, metadata = get_config_with_metadata()

        typer.echo("Configuration Check:")

        # Get the config source info from the first field's metadata
        # All fields should come from the same source file
        config_file = None
        config_source = None

        if metadata.per_field:
            # Get the first field's source info (they should all be the same)
            first_field_source = next(iter(metadata.per_field.values()))
            config_file = first_field_source.source_path
            config_source = first_field_source.source

        if config_file and config_source != "defaults":
            config_path = Path(config_file)
            typer.secho(
                f"✓ External config found: {config_path}",
                fg=typer.colors.GREEN,
            )
            typer.echo(f"  Source: {config_source}")
        else:
            typer.secho(
                "⚠ No external config found, using defaults",
                fg=typer.colors.YELLOW,
            )

    except Exception as e:
        typer.secho(
            f"✗ Error checking configuration: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        logger.exception("Configuration check failed")
        raise typer.Exit(1) from e
