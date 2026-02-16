"""Voice recording commands (manual and automatic VAD-based)."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from voicepad_core import (
    Config,
    get_config,
    get_device_by_index,
    get_input_devices,
    record_voice,
)

logger = logging.getLogger(__name__)

recording_app = typer.Typer(help="Voice recording commands")


def _resolve_device_index(device_index: int | None, config: Config) -> int:
    if device_index is not None:
        return device_index
    if config.input_device_index is not None:
        return config.input_device_index
    return 0


@recording_app.command()
def manual(
    device_index: Annotated[
        int | None,
        typer.Option(
            "--device-index",
            "-d",
            help="Audio device index (leave empty to use config or first device)",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save recording (default: data/recordings)",
        ),
    ] = None,
) -> None:
    """Record audio continuously until Enter is pressed."""
    try:
        devices = get_input_devices()
        if not devices:
            typer.secho("✗ No audio input devices found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        config = get_config()
        resolved_index = _resolve_device_index(device_index, config)

        # Validate device exists
        dev = get_device_by_index(resolved_index)
        typer.echo(f"Recording from device {resolved_index}: {dev.name}")
        typer.echo(f"Channels: {dev.channels}, Sample rate: {dev.sample_rate}Hz")
        typer.echo("Press Enter to stop recording...")

        # Use default output directory if not specified
        if output_dir is None:
            output_dir = Path("data/recordings")

        # Record audio
        output_path = record_voice(resolved_index, output_dir)

        typer.secho(f"✓ Recording completed: {output_path}", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
