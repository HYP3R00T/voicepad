"""Audio recording CLI commands for voicepad."""

import logging
import signal
import sys
import time

import typer
from voicepad_core import AudioRecorder, AudioRecorderError, get_config

logger = logging.getLogger(__name__)

record_app = typer.Typer(help="Audio recording commands")


def _handle_interrupt(recorder: AudioRecorder) -> None:
    """Handle keyboard interrupt during recording.

    Args:
        recorder: The AudioRecorder instance to stop.
    """
    typer.echo("\n\n[!] Recording stopped by user")
    try:
        output_file = recorder.stop_recording()
        if output_file:
            typer.secho(f"[OK] Recording saved to: {output_file}", fg=typer.colors.GREEN)
    except AudioRecorderError as e:
        typer.secho(f"[ERROR] Error stopping recording: {e}", fg=typer.colors.RED, err=True)


@record_app.command("start")
def start_recording(
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Custom prefix for the recording filename (overrides config)",
    ),
    duration: float | None = typer.Option(
        None,
        "--duration",
        "-d",
        help="Duration in seconds for fixed-length recording (optional)",
        min=0.1,
    ),
) -> None:
    """Start recording audio from the configured input device.

    The recording will use the configured microphone (input_device_index) and
    save to the configured recordings directory (recordings_path).

    Press Ctrl+C to stop recording manually, or use --duration for automatic stop.
    """
    try:
        # Load configuration
        config = get_config()

        # Display configuration info
        typer.echo("Recording Configuration:")
        typer.echo(f"   Input device: {config.input_device_index or 'default'}")
        typer.echo(f"   Output directory: {config.recordings_path}")
        typer.echo(f"   Filename prefix: {prefix or config.recording_prefix}")
        typer.echo()

        # Create recorder
        recorder = AudioRecorder(config)

        # Start recording
        output_file = recorder.start_recording(prefix=prefix, duration=duration)

        if duration:
            # Fixed-duration recording
            typer.secho(f"[REC] Recording for {duration} seconds...", fg=typer.colors.YELLOW)
            typer.echo(f"   Output: {output_file}")

            # Wait for recording to complete
            time.sleep(duration + 0.5)  # Add small buffer for processing

            typer.secho("[OK] Recording completed!", fg=typer.colors.GREEN)
            typer.echo(f"   Saved to: {output_file}")

        else:
            # Manual stop recording
            typer.secho("[REC] Recording in progress...", fg=typer.colors.YELLOW)
            typer.echo(f"   Output: {output_file}")
            typer.echo()
            typer.echo("Press Ctrl+C to stop recording")

            # Set up signal handler for graceful shutdown
            def signal_handler(sig: int, frame) -> None:
                _handle_interrupt(recorder)
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            # Keep the main thread alive while recording
            try:
                while recorder.is_recording():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                _handle_interrupt(recorder)

    except AudioRecorderError as e:
        typer.secho(f"[ERROR] Recording error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        logger.exception("Unexpected error during recording")
        typer.secho(f"[ERROR] Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e


@record_app.command("info")
def show_info() -> None:
    """Display current recording configuration and status."""
    try:
        config = get_config()

        typer.echo("Current Recording Configuration")
        typer.echo("=" * 60)
        typer.echo(f"Input device index: {config.input_device_index or 'default (system)'}")
        typer.echo(f"Recordings directory: {config.recordings_path}")
        typer.echo(f"Filename prefix: {config.recording_prefix}")
        typer.echo("=" * 60)

        # Check if recordings directory exists
        if config.recordings_path.exists():
            typer.secho("[OK] Recordings directory exists", fg=typer.colors.GREEN)

            # Count existing recordings
            recordings = list(config.recordings_path.glob("*.wav"))
            typer.echo(f"   {len(recordings)} recording(s) found")
        else:
            typer.secho("[WARN] Recordings directory does not exist (will be created)", fg=typer.colors.YELLOW)

        typer.echo()
        typer.echo("Tip: Use 'voicepad config input' to view and configure audio devices")

    except Exception as e:
        typer.secho(f"[ERROR] Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
