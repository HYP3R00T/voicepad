"""CLI commands for voice recording and transcription."""

import logging
import os
import threading
from pathlib import Path
from typing import Annotated

import typer

from voicepad.config.settings import SUPPORTED_MODEL_SIZES, get_config
from voicepad.system_utils import check_gpu_capabilities, recommend_faster_whisper_model
from voicepad.voice.recorder import print_devices, record_voice
from voicepad.voice.transcriber import TranscriptionPoller, TranscriptionResult, transcribe_audio
from voicepad.voice.utils import get_recording_path

logger = logging.getLogger(__name__)

voice_app = typer.Typer(help="Voice recording and transcription commands")


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


@voice_app.command()
def list_devices() -> None:
    """List available audio input devices."""
    print_devices()


@voice_app.command()
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


@voice_app.command()
def transcribe(
    audio_file: Annotated[
        Path,
        typer.Argument(
            help="Path to audio file to transcribe",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Whisper model size (tiny, base, small, medium, large-v2, large-v3, turbo). Uses config if not specified.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file for transcription (default: same as audio with .txt extension)",
        ),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            "-s",
            help="Stream transcription output in real-time as it processes",
        ),
    ] = False,
) -> None:
    """Transcribe audio file to text using faster-whisper."""
    config = get_config()
    model = _resolve_model(model, config)

    typer.echo(f"Transcribing: {audio_file}")
    typer.echo("\n⏳ Processing audio...")

    # Suppress HuggingFace symlinks warning
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        # Prepare segment callback for streaming
        transcribed_text = []

        def on_segment_callback(text: str) -> None:
            """Callback for each transcribed segment."""
            transcribed_text.append(text)
            if stream:
                typer.echo(text, nl=False)

        # Transcribe audio
        result: TranscriptionResult = transcribe_audio(
            audio_path=audio_file,
            model_size=model,
            on_segment=on_segment_callback,
        )

        # Show transcription text first (either already streamed or display now)
        if not stream:
            typer.echo("\nTranscription:")
            typer.echo("-" * 60)
            typer.echo(result.text)
            typer.echo("-" * 60)

        # Processing complete message
        if stream:
            typer.echo("\n\n✓ Processing complete!")
        else:
            typer.echo("\n✓ Processing complete!")

        # Display metadata summary
        typer.echo("\n" + "=" * 60)
        typer.secho("Transcription Complete", fg=typer.colors.GREEN, bold=True)
        typer.echo("=" * 60)
        typer.echo(f"Detected language: {result.language}")
        typer.echo(f"Duration: {result.duration:.2f}s")
        typer.echo(f"Segments: {len(result.segments)}")
        typer.echo(f"Device: {result.device_used}")
        typer.echo("=" * 60)

        # Save to file - use config markdown path if output not specified
        if output is None:
            filename = audio_file.stem + ".txt"
            markdown_dir = config.markdown_path
            markdown_dir.mkdir(parents=True, exist_ok=True)
            output = markdown_dir / filename

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.text, encoding="utf-8")
        typer.secho(f"\n✓ Saved to: {output}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"✗ Transcription failed: {e}", fg=typer.colors.RED, err=True)
        logger.error("Transcription failed", exc_info=e)
        raise typer.Exit(1) from e


@voice_app.command()
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
