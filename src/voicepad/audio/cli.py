"""CLI commands for audio recording and device management."""

import logging

import typer

from voicepad.audio.scanner import print_devices, record_voice

logger = logging.getLogger(__name__)

audio_app = typer.Typer(help="Audio recording and device commands")


@audio_app.command()
def list_devices() -> None:
    """List available audio input devices."""
    print_devices()


@audio_app.command()
def record(
    device_index: int = typer.Option(0, help="Audio device index"),
) -> None:
    """Record audio continuously until Enter is pressed."""
    try:
        typer.echo(f"Recording from device {device_index}...")
        typer.echo("Press Enter to stop recording...")
        record_voice(device_index)
        typer.echo("✓ Recording completed")
    except ValueError as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1) from e
