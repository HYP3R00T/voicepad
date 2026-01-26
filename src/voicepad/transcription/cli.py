"""CLI commands for audio transcription."""

import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from voicepad.config.settings import SUPPORTED_MODEL_SIZES, get_config
from voicepad.system_utils import check_gpu_capabilities, recommend_faster_whisper_model
from voicepad.transcription import TranscriptionResult, transcribe_audio

logger = logging.getLogger(__name__)

transcription_app = typer.Typer(help="Audio transcription commands")


@transcription_app.command()
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
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code (None for auto-detection)",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            "-d",
            help="Device to use (cuda, cpu, auto). Uses config if not specified.",
        ),
    ] = None,
    compute_type: Annotated[
        str | None,
        typer.Option(
            "--compute-type",
            "-c",
            help="Compute type (float16, int8, int8_float16, auto). Uses config if not specified.",
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
    # Load config for defaults
    config = get_config()

    # Use config values if not specified via CLI
    if model is None:
        if config.transcription.model == "auto":
            # Auto-detect best model based on GPU
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

    typer.echo(f"Transcribing: {audio_file}")
    typer.echo("\n⏳ Processing audio...")

    # Suppress HuggingFace symlinks warning
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        # Show progress message
        typer.echo("\n⏳ Processing audio...")

        # Prepare segment callback for streaming
        transcribed_text = []

        def on_segment_callback(text: str) -> None:
            """Callback for each transcribed segment."""
            transcribed_text.append(text)
            if stream:
                # Print segment without newline for continuous text
                typer.echo(text, nl=False)
                # Flush to ensure immediate display

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
        typer.echo(f"Language: {result.language}")
        typer.echo(f"Duration: {result.duration:.2f}s")
        typer.echo(f"Segments: {len(result.segments)}")
        typer.echo(f"Device: {result.device_used}")
        typer.echo("=" * 60)

        # Save to file - use config markdown path if output not specified
        if output is None:
            # Create filename from audio file name
            filename = audio_file.stem + ".txt"
            # Use markdown_path from config
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
