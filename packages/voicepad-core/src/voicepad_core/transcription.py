"""Audio transcription using faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Base exception for transcription errors."""


def resolve_auto_settings(config: Config) -> tuple[str, str]:
    """Resolve 'auto' device and compute_type to actual values.

    Args:
        config: Configuration object with transcription settings.

    Returns:
        Tuple of (device, compute_type) resolved to actual values.
        Example: ("cuda", "float16") or ("cpu", "int8")
    """
    device = config.transcription_device
    compute_type = config.transcription_compute_type

    # If either is auto, run system detection
    if device == "auto" or compute_type == "auto":
        from voicepad_core import (
            get_available_models,
            get_cpu_info,
            get_model_recommendation,
            get_ram_info,
            gpu_diagnostics,
        )
        from voicepad_core.diagnostics.models import SystemInfo

        logger.info("Resolving auto transcription settings via system detection")

        # Gather system info
        ram = get_ram_info()
        cpu = get_cpu_info()
        gpu = gpu_diagnostics()
        available_models = get_available_models()
        system_info = SystemInfo(ram=ram, cpu=cpu, gpu_diagnostics=gpu)

        # Get recommendation
        rec = get_model_recommendation(system_info, available_models)

        if device == "auto":
            device = rec.recommended_device
            logger.info(f"Resolved device: {device}")

        if compute_type == "auto":
            compute_type = rec.recommended_compute_type
            logger.info(f"Resolved compute_type: {compute_type}")

    return device, compute_type


def format_transcription_markdown(
    audio_path: Path,
    segments: list[Any],
    info: Any,
    config: Config,
    device: str,
    compute_type: str,
) -> str:
    """Format transcription results as Markdown.

    Args:
        audio_path: Path to the audio file that was transcribed.
        segments: List of transcription segments with timestamps.
        info: Transcription info object with metadata.
        config: Configuration object.
        device: Device used for transcription.
        compute_type: Compute type used for transcription.

    Returns:
        Formatted Markdown string.
    """
    lines = [
        "# Transcription",
        "",
        f"**File:** {audio_path.name}",
        f"**Model:** {config.transcription_model}",
        f"**Device:** {device}",
        f"**Compute Type:** {compute_type}",
        f"**Language:** {info.language} ({info.language_probability * 100:.1f}% confidence)",
        f"**Duration:** {info.duration:.1f} seconds",
        "",
        "---",
        "",
        "## Text",
        "",
    ]

    # Add segments with timestamps
    for segment in segments:
        timestamp = f"[{segment.start:.1f} -> {segment.end:.1f}]"
        text = segment.text.strip()
        lines.append(f"{timestamp} {text}")

    return "\n".join(lines)


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    config: Config,
) -> dict[str, Any]:
    """Transcribe audio file and save as Markdown.

    Args:
        audio_path: Path to the audio file to transcribe.
        output_path: Path where the Markdown transcription will be saved.
        config: Configuration object with transcription settings.

    Returns:
        Dictionary with transcription statistics:
        - language: Detected language code
        - language_probability: Confidence of language detection
        - duration: Audio duration in seconds
        - word_count: Approximate word count
        - segment_count: Number of segments

    Raises:
        TranscriptionError: If transcription fails.
    """
    # Validate input file
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise TranscriptionError(f"Audio path is not a file: {audio_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Resolve auto settings
        device, compute_type = resolve_auto_settings(config)

        logger.info(f"Loading Whisper model: {config.transcription_model}")
        logger.info(f"Device: {device}, Compute Type: {compute_type}")

        # Load the Whisper model
        try:
            model = WhisperModel(
                config.transcription_model,
                device=device,
                compute_type=compute_type,
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to load model '{config.transcription_model}': {e}") from e

        # Transcribe the audio
        logger.info(f"Transcribing: {audio_path}")
        try:
            segments_iter, info = model.transcribe(str(audio_path), beam_size=5)
            # Convert generator to list so we can iterate multiple times
            segments = list(segments_iter)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        logger.info(f"Detected language: {info.language} ({info.language_probability * 100:.1f}%)")
        logger.info(f"Duration: {info.duration:.1f}s")

        # Format as Markdown
        markdown_content = format_transcription_markdown(
            audio_path=audio_path,
            segments=segments,
            info=info,
            config=config,
            device=device,
            compute_type=compute_type,
        )

        # Save to file
        try:
            output_path.write_text(markdown_content, encoding="utf-8")
            logger.info(f"Transcription saved to: {output_path}")
        except Exception as e:
            raise TranscriptionError(f"Failed to save transcription: {e}") from e

        # Calculate statistics
        word_count = sum(len(segment.text.split()) for segment in segments)

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "word_count": word_count,
            "segment_count": len(segments),
        }

    except TranscriptionError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {e}")
        raise TranscriptionError(f"Unexpected error: {e}") from e
