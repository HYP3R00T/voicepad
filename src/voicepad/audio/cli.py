"""CLI commands for audio recording and device management."""

import logging
from pathlib import Path

import typer

from voicepad.audio.scanner import print_devices, record_voice, record_voice_continuous
from voicepad.audio.utils import get_recording_path
from voicepad.config import get_config

logger = logging.getLogger(__name__)

audio_app = typer.Typer(help="Audio recording and device commands")


@audio_app.command()
def list_devices() -> None:
    """List available audio input devices."""
    print_devices()


@audio_app.command()
def record(
    device_index: int = typer.Option(0, help="Audio device index"),
    duration: float = typer.Option(5.0, help="Recording duration in seconds"),
    output: str | None = typer.Option(None, help="Output file path (default: config path)"),
) -> None:
    """Record audio for a specified duration."""
    try:
        typer.echo(f"Recording from device {device_index} for {duration} seconds...")
        data = record_voice(device_index, duration)

        if output:
            Path(output).write_bytes(data)
            typer.echo(f"✓ Recording saved to: {output}")
        else:
            config = get_config()
            saved_path = get_recording_path(config.recordings_path)
            typer.echo(f"✓ Recording saved to: {saved_path}")
    except ValueError as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1) from e


@audio_app.command()
def record_continuous(
    device_index: int = typer.Option(0, help="Audio device index"),
) -> None:
    """Record audio continuously until Enter is pressed."""
    try:
        typer.echo(f"Recording from device {device_index}...")
        typer.echo("Press Enter to stop recording...")
        record_voice_continuous(device_index)
        typer.echo("✓ Recording completed")
    except ValueError as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1) from e
