"""Configuration management commands."""

import logging

import typer
from voicepad_core import get_config, get_config_with_metadata, get_input_devices

logger = logging.getLogger(__name__)

config_app = typer.Typer(help="Configuration management commands")


@config_app.command("input")
def list_input_devices() -> None:
    """List available audio input devices for configuration."""
    devices = get_input_devices()
    if not devices:
        typer.secho("No audio input devices detected.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    config = get_config()
    configured_index = config.input_device_index

    typer.echo("Configured default input device:")
    if configured_index is None:
        typer.echo("  not set")
    else:
        try:
            configured_device = next(dev for dev in devices if dev.index == configured_index)
            typer.echo(f"  {configured_index}: {configured_device.name}")
        except StopIteration:
            typer.echo(f"  {configured_index}: not found")

    typer.echo("\nAvailable audio input devices:")
    typer.echo("-" * 60)
    for dev in devices:
        selected_marker = " (configured)" if dev.index == configured_index else ""
        typer.echo(f"{dev}{selected_marker}")
    typer.echo("-" * 60)

    _show_config_hint()


def _show_config_hint() -> None:
    config_file = None
    config_source = None

    _config, metadata = get_config_with_metadata()
    if metadata.per_field:
        first_field_source = next(iter(metadata.per_field.values()))
        config_file = first_field_source.source_path
        config_source = first_field_source.source

    if config_file and config_source != "defaults":
        typer.echo(f"Config file: {config_file}")
        typer.echo("Set input_device_index in that file to persist the default.")
    else:
        typer.echo("Config file: using defaults (no external config found)")
        typer.echo("Set input_device_index in voicepad.yaml to persist the default.")
