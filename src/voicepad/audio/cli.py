"""CLI commands for audio recording and device management."""

import logging
import os
import threading
from typing import Annotated

import typer

from voicepad.audio.scanner import get_recording_path, print_devices, record_voice
from voicepad.config.settings import SUPPORTED_MODEL_SIZES, get_config
from voicepad.system_utils import check_gpu_capabilities, recommend_faster_whisper_model
from voicepad.transcription.transcriber import TranscriptionPoller

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


@audio_app.command()
def record_and_transcribe(
    device_index: Annotated[
        int,
        typer.Option(
            "--device",
            "-d",
            help="Audio device index to record from",
        ),
    ] = 0,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Whisper model size. Uses config if not specified.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code (None for auto-detection)",
        ),
    ] = None,
    device_type: Annotated[
        str | None,
        typer.Option(
            "--compute-device",
            help="Device for transcription (cuda, cpu, auto). Uses config if not specified.",
        ),
    ] = None,
    poll_interval: Annotated[
        float,
        typer.Option(
            "--poll-interval",
            help="Seconds between transcription polls (default: 30)",
        ),
    ] = 30.0,
) -> None:
    """Record audio and transcribe in real-time.

    Records continuously while polling the temp audio file to transcribe accumulated audio
    at regular intervals. Press Enter to stop recording.
    """
    # Load config for defaults
    config = get_config()

    # Determine model
    if model is None:
        if config.transcription.model == "auto":
            gpu_info = check_gpu_capabilities()
            rec = recommend_faster_whisper_model(gpu_info.device_type, gpu_info.total_memory_gb)
            model = rec.model_size
            typer.echo(
                f"Auto-detected model: {model} (based on {gpu_info.device_type} with {gpu_info.total_memory_gb or 0:.1f}GB VRAM)"
            )
        else:
            model = config.transcription.model
            if model not in SUPPORTED_MODEL_SIZES:
                typer.echo(
                    "✗ Unsupported model in config. Supported: "
                    + ", ".join(m for m in SUPPORTED_MODEL_SIZES if m != "auto"),
                    err=True,
                )
                raise typer.Exit(1)
            typer.echo(f"Using configured model: {model}")
    else:
        if model not in SUPPORTED_MODEL_SIZES:
            typer.echo(
                "✗ Unsupported model. Supported: " + ", ".join(m for m in SUPPORTED_MODEL_SIZES if m != "auto"),
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Using specified model: {model}")

    if device_type is None:
        device_type = config.transcription.device

    if language is None:
        language = config.transcription.language

    typer.echo(f"Recording from device {device_index}...")
    typer.echo("Press Enter to stop recording...")
    typer.echo(f"Device: {device_type}, Model: {model}, Poll interval: {poll_interval}s")
    typer.echo("")  # Empty line before transcription starts

    # Suppress HuggingFace warnings
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Determine recording output path from config
    config.recordings_path.mkdir(parents=True, exist_ok=True)
    temp_audio_file = get_recording_path(config.recordings_path)

    try:
        # Create transcription poller
        poller = TranscriptionPoller(
            model_size=model,
            device=device_type,  # type: ignore[arg-type]
            compute_type=config.transcription.compute_type,  # type: ignore[arg-type]
            language=language,
            poll_interval=poll_interval,
            min_duration=3.0,
        )

        # Collect segments during recording instead of printing
        collected_segments: list[str] = []

        def on_segment_callback(text: str) -> None:
            """Callback for each new segment - collect it."""
            collected_segments.append(text)

        # Stop event for threading
        stop_event = threading.Event()

        # Thread for polling and transcribing
        def polling_loop() -> None:
            """Poll file and transcribe in background."""
            while not stop_event.is_set():
                try:
                    result = poller.poll_and_transcribe(
                        temp_audio_file,
                        on_segment=on_segment_callback,
                        stop_event=stop_event,
                    )
                    if result:
                        logger.debug(f"Polled transcription: {len(result.segments)} segments")
                except Exception as e:
                    logger.error(f"Polling error: {e}")

                # Sleep before next poll
                stop_event.wait(poll_interval)

        # Start polling thread
        poll_thread = threading.Thread(target=polling_loop, daemon=True)
        poll_thread.start()

        # Record audio (blocks until Enter pressed)
        _, output_path = record_voice(device_index, output_file=temp_audio_file)

        # Signal stop to polling thread
        stop_event.set()
        poll_thread.join(timeout=5)

        # Final transcription pass to get any remaining audio
        final_result = poller.poll_and_transcribe(
            temp_audio_file,
            on_segment=on_segment_callback,
            stop_event=stop_event,
        )

        # Display all collected transcription after recording stops
        typer.echo("\n")
        if collected_segments:
            typer.echo("".join(collected_segments))
        typer.echo("\n")

        # Save transcript file
        if output_path and final_result:
            markdown_dir = config.markdown_path
            markdown_dir.mkdir(parents=True, exist_ok=True)
            markdown_file = markdown_dir / f"{output_path.stem}_transcript.txt"
            markdown_file.write_text(final_result.text, encoding="utf-8")
            typer.echo(f"✓ Transcript: {markdown_file}")

        # Show where audio was saved
        typer.echo(f"✓ Audio: {output_path}")

    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        logger.error("Record and transcribe failed", exc_info=e)
        raise typer.Exit(1) from e
    finally:
        # Do not delete recorded audio; stored in configured recordings path
        pass
