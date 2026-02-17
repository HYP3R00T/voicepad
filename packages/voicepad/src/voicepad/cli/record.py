"""Audio recording CLI commands for voicepad."""

import logging
import signal
import time
from pathlib import Path

import typer
from voicepad_core import AudioRecorder, AudioRecorderError, get_config

logger = logging.getLogger(__name__)

record_app = typer.Typer(help="Audio recording commands")


def _handle_interrupt(recorder: AudioRecorder) -> Path | None:
    """Handle keyboard interrupt during recording.

    Args:
        recorder: The AudioRecorder instance to stop.

    Returns:
        Path to saved audio file, or None if failed.
    """
    typer.echo("\n\n[!] Recording stopped by user")
    try:
        output_file = recorder.stop_recording()
        if output_file:
            typer.secho(f"[OK] Recording saved to: {output_file}", fg=typer.colors.GREEN)
            return output_file
    except AudioRecorderError as e:
        typer.secho(f"[ERROR] Error stopping recording: {e}", fg=typer.colors.RED, err=True)
    return None


def _transcribe_audio_file(output_file: Path, config) -> None:
    """Transcribe audio file and display results.

    Args:
        output_file: Path to the audio file to transcribe.
        config: Configuration object.
    """
    typer.echo()
    typer.secho("[*] Transcribing audio...", fg=typer.colors.CYAN)

    try:
        from voicepad_core import TranscriptionError, transcribe_audio

        # Generate output path
        md_path = config.markdown_path / f"{output_file.stem}.md"
        config.markdown_path.mkdir(parents=True, exist_ok=True)

        # Transcribe
        stats = transcribe_audio(output_file, md_path, config)

        # Check if GPU fallback occurred
        fallback_info = stats.get("fallback_info", {})
        if fallback_info.get("fallback_occurred"):
            typer.echo()
            typer.secho("[!] GPU requested but CUDA libraries not available", fg=typer.colors.YELLOW)

            # Show what's missing
            missing = fallback_info.get("missing_components", [])
            if missing:
                typer.echo(f"  Missing: {', '.join(missing)}")

            typer.echo()
            typer.echo("  For 4x faster transcription, install GPU support:")
            typer.secho("    pip install voicepad-core[gpu]", fg=typer.colors.CYAN)
            typer.echo()
            typer.echo("  Using CPU for now...")

        # Success message
        typer.echo()
        device = stats.get("device", "cpu")
        duration = stats.get("duration", 0)
        typer.secho(f"[OK] Transcription saved! ({duration:.1f}s using {device})", fg=typer.colors.GREEN)
        typer.echo(f"   File: {md_path}")
        typer.echo(f"   Language: {stats['language']} ({stats['language_probability'] * 100:.1f}%)")
        typer.echo(f"   Words: {stats['word_count']}")
        typer.echo(f"   Segments: {stats['segment_count']}")

    except TranscriptionError as e:
        typer.secho(f"[ERROR] Transcription failed: {e}", fg=typer.colors.RED, err=True)
        typer.echo("   Recording was saved successfully, but transcription failed.")
    except Exception as e:
        logger.exception("Unexpected error during transcription")
        typer.secho(f"[ERROR] Unexpected transcription error: {e}", fg=typer.colors.RED, err=True)


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
    transcribe: bool = typer.Option(
        True,
        "--transcribe/--no-transcribe",
        help="Transcribe audio after recording (default: enabled)",
    ),
) -> None:
    """Start recording audio from the configured input device.

    The recording will use the configured microphone (input_device_index) and
    save to the configured recordings directory (recordings_path).

    By default, the audio will be automatically transcribed after recording stops.
    Use --no-transcribe to skip transcription.

    Press Ctrl+C to stop recording manually, or use --duration for automatic stop.
    """
    output_file: Path | None = None

    try:
        # Load configuration
        config = get_config()

        # Display configuration info
        typer.echo("Recording Configuration:")
        typer.echo(f"   Input device: {config.input_device_index or 'default'}")
        typer.echo(f"   Output directory: {config.recordings_path}")
        typer.echo(f"   Filename prefix: {prefix or config.recording_prefix}")
        if transcribe:
            typer.echo(f"   Transcription: enabled (model: {config.transcription_model})")
        else:
            typer.echo("   Transcription: disabled")
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
                nonlocal output_file
                output_file = _handle_interrupt(recorder)
                # Don't exit here - let the finally block handle transcription

            signal.signal(signal.SIGINT, signal_handler)

            # Keep the main thread alive while recording
            try:
                while recorder.is_recording():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                output_file = _handle_interrupt(recorder)

    except AudioRecorderError as e:
        typer.secho(f"[ERROR] Recording error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        logger.exception("Unexpected error during recording")
        typer.secho(f"[ERROR] Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    finally:
        # Transcribe if enabled and we have a valid output file
        if transcribe and output_file and output_file.exists():
            _transcribe_audio_file(output_file, config)


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
