import logging
from pathlib import Path
from typing import Annotated

import typer
from voicepad_core import get_device_by_index, get_input_devices, record_voice

logger = logging.getLogger(__name__)

voice_app = typer.Typer(help="Voice recording and audio device commands")


@voice_app.command()
def list_devices() -> None:
    """List available audio input devices."""
    devices = get_input_devices()
    if not devices:
        typer.echo("No audio input devices found.", err=True)
        raise typer.Exit(1)

    typer.echo("Available audio input devices:")
    typer.echo("-" * 60)
    for dev in devices:
        typer.echo(str(dev))
    typer.echo("-" * 60)


@voice_app.command()
def record(
    device_index: Annotated[
        int,
        typer.Option(
            "--device-index",
            "-d",
            help="Audio device index",
        ),
    ] = 0,
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
        # Validate device exists
        dev = get_device_by_index(device_index)
        typer.echo(f"Recording from device {device_index}: {dev.name}")
        typer.echo(f"Channels: {dev.channels}, Sample rate: {dev.sample_rate}Hz")
        typer.echo("Press Enter to stop recording...")

        # Use default output directory if not specified
        if output_dir is None:
            output_dir = Path("data/recordings")

        # Record audio
        output_path = record_voice(device_index, output_dir)

        typer.secho(f"✓ Recording completed: {output_path}", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
