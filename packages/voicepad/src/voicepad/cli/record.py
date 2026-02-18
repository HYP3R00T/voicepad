"""Audio recording CLI commands for voicepad."""

import logging
import sys
import threading
import time
from pathlib import Path

import typer
from voicepad_core import AudioRecorder, AudioRecorderError, get_config

logger = logging.getLogger(__name__)

record_app = typer.Typer(help="Audio recording commands")


def _stop_recording(recorder: AudioRecorder) -> Path | None:
    """Stop the active recording and report the result.

    Args:
        recorder: The AudioRecorder instance to stop.

    Returns:
        Path to saved audio file, or None if failed.
    """
    typer.echo("\n[!] Stopping recording...")
    try:
        output_file = recorder.stop_recording()
        if output_file:
            typer.secho(f"[OK] Recording saved to: {output_file}", fg=typer.colors.GREEN)
            return output_file
    except AudioRecorderError as e:
        typer.secho(f"[ERROR] Error stopping recording: {e}", fg=typer.colors.RED, err=True)
    return None


def _wait_for_quit_key(stop_event: threading.Event) -> None:
    """Block until the user types 'q' and presses Enter.

    Runs in a daemon thread. Sets stop_event when 'q' is received.
    Any other input is silently ignored, so accidental Enter presses
    do not stop the recording.

    The stop_event is a plain threading.Event — future frontends (TUI,
    web) can set it from a button handler or API call without touching
    this function or voicepad-core.

    Args:
        stop_event: Event to set when the user requests a stop.
    """
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                stop_event.set()
                break
        except (EOFError, OSError):
            # stdin closed (e.g. piped input or UV tool wrapper edge case)
            stop_event.set()
            break


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
    vad: bool | None = typer.Option(
        None,
        "--vad/--no-vad",
        help="Enable VAD-based chunking for long recordings (uses config default if not specified)",
    ),
    min_chunk_duration: float | None = typer.Option(
        None,
        "--min-chunk-duration",
        help="Minimum chunk duration in seconds (VAD mode only, default: 60)",
        min=10.0,
        max=600.0,
    ),
    vad_threshold: float | None = typer.Option(
        None,
        "--vad-threshold",
        help="VAD speech detection threshold 0.0-1.0 (VAD mode only, default: 0.5)",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Start recording audio from the configured input device.

    The recording will use the configured microphone (input_device_index) and
    save to the configured recordings directory (recordings_path).

    By default, the audio will be automatically transcribed after recording stops.
    Use --no-transcribe to skip transcription.

    Type q and press Enter to stop recording manually, or use --duration for automatic stop.
    """
    output_file: Path | None = None

    try:
        # Load configuration
        config = get_config()

        # Apply CLI overrides for VAD settings
        if vad is not None or min_chunk_duration is not None or vad_threshold is not None:
            # Create modified config with CLI overrides
            config_dict = config.model_dump()

            if vad is not None:
                config_dict["vad_enabled"] = vad
            if min_chunk_duration is not None:
                config_dict["vad_min_chunk_duration"] = min_chunk_duration
            if vad_threshold is not None:
                config_dict["vad_threshold"] = vad_threshold

            # Reconstruct config with overrides
            from voicepad_core.config import Config

            config = Config(**config_dict)

        # Display configuration info
        typer.echo("Recording Configuration:")
        typer.echo(f"   Input device: {config.input_device_index or 'default'}")
        typer.echo(f"   Output directory: {config.recordings_path}")
        typer.echo(f"   Filename prefix: {prefix or config.recording_prefix}")

        if config.vad_enabled:
            typer.echo("   VAD chunking: enabled")
            typer.echo(f"      Min chunk duration: {config.vad_min_chunk_duration}s")
            typer.echo(f"      VAD threshold: {config.vad_threshold}")
            typer.echo(f"      Min silence: {config.vad_min_silence_duration_ms}ms")
        else:
            typer.echo("   VAD chunking: disabled")

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
            typer.echo("Type q and press Enter to stop recording")

            # stop_event is the loose-coupling bridge:
            # the CLI sets it via stdin; a TUI or web frontend can set it
            # from a button handler or API call — no core changes needed.
            stop_event = threading.Event()

            listener = threading.Thread(
                target=_wait_for_quit_key,
                args=(stop_event,),
                daemon=True,
            )
            listener.start()

            # Block main thread until q is typed or recording ends externally
            while recorder.is_recording() and not stop_event.is_set():
                time.sleep(0.05)

            if stop_event.is_set():
                output_file = _stop_recording(recorder)

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
            # For VAD mode, transcription already happened in background
            # For non-VAD mode, transcribe now
            if config.vad_enabled:
                # Transcription already complete in background
                typer.echo()
                typer.secho("✓ Transcription already complete (background processing)", fg=typer.colors.GREEN)
                md_path = output_file.with_suffix(".md")
                if md_path.exists():
                    typer.echo(f"   Markdown: {md_path}")
            else:
                # Non-VAD mode: transcribe now
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
        if config.vad_enabled:
            typer.echo("VAD chunking: enabled")
            typer.echo(f"   Min chunk duration: {config.vad_min_chunk_duration}s")
            typer.echo(f"   VAD threshold: {config.vad_threshold}")
            typer.echo(f"   Min silence: {config.vad_min_silence_duration_ms}ms")
        else:
            typer.echo("VAD chunking: disabled")
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
