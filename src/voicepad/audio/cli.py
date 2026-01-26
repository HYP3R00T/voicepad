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


def _resolve_model(model: str | None, config) -> str:
    """Resolve which model to use given CLI value and config."""
    supported = ", ".join(m for m in SUPPORTED_MODEL_SIZES if m != "auto")

    def _validate(value: str) -> str:
        if value not in SUPPORTED_MODEL_SIZES:
            typer.echo(f"✗ Unsupported model. Supported: {supported}", err=True)
            raise typer.Exit(1)
        return value

    if model:
        typer.echo(f"Using specified model: {model}")
        return _validate(model)

    if config.transcription.model == "auto":
        gpu_info = check_gpu_capabilities()
        rec = recommend_faster_whisper_model(gpu_info.device_type, gpu_info.total_memory_gb)
        detected = _validate(rec.model_size)
        typer.echo(
            f"Auto-detected model: {detected} (based on {gpu_info.device_type} with {gpu_info.total_memory_gb or 0:.1f}GB VRAM)"
        )
        return detected

    configured = _validate(config.transcription.model)
    typer.echo(f"Using configured model: {configured}")
    return configured


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
    config = get_config()
    model = _resolve_model(model, config)

    typer.echo(f"Recording from device {device_index}...")
    typer.echo("Press Enter to stop recording...")
    typer.echo(f"Model: {model}, Poll interval: {poll_interval}s")
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
            poll_interval=poll_interval,
            min_duration=3.0,
        )

        # Collect segments during recording
        collected_segments: list[str] = []
        stop_event = threading.Event()

        def polling_loop() -> None:
            """Poll file and transcribe in background."""
            append_segment = collected_segments.append
            while not stop_event.is_set():
                try:
                    poller.poll_and_transcribe(
                        temp_audio_file,
                        on_segment=append_segment,
                        stop_event=stop_event,
                    )
                except Exception as e:
                    logger.error(f"Polling error: {e}")

                stop_event.wait(poll_interval)

        # Start polling thread
        poll_thread = threading.Thread(target=polling_loop, daemon=True)
        poll_thread.start()

        # Record audio (blocks until Enter pressed)
        output_path = record_voice(device_index, output_file=temp_audio_file)

        # Stop polling and do final transcription pass
        stop_event.set()
        poll_thread.join(timeout=5)
        append_segment = collected_segments.append
        final_result = poller.poll_and_transcribe(
            temp_audio_file,
            on_segment=append_segment,
            stop_event=stop_event,
        )

        # Display all transcription
        if collected_segments:
            typer.echo("\n" + "".join(collected_segments) + "\n")

        # Save transcript and show file paths
        if output_path and final_result:
            markdown_dir = config.markdown_path
            markdown_dir.mkdir(parents=True, exist_ok=True)
            markdown_file = markdown_dir / f"{output_path.stem}_transcript.txt"
            markdown_file.write_text(final_result.text, encoding="utf-8")
            typer.echo(f"✓ Transcript: {markdown_file}")
        typer.echo(f"✓ Audio: {output_path}")

    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        logger.error("Record and transcribe failed", exc_info=e)
        raise typer.Exit(1) from e
