"""Core transcription functionality using faster-whisper."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "cpu", "auto"]
ComputeType = Literal["int8", "int8_float16", "int16", "float16", "float32"]


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: str
    segments: list[dict]
    duration: float
    device_used: str


class WhisperTranscriber:
    """Wrapper for faster-whisper transcription with GPU support."""

    def __init__(
        self,
        model_size: str = "base",
        device: DeviceType = "auto",
        compute_type: ComputeType = "float16",
    ) -> None:
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (cuda, cpu, auto)
            compute_type: Compute type for model inference
        """
        self.model_size = model_size
        self.device = self._determine_device(device)
        self.compute_type = self._determine_compute_type(compute_type)
        self.model: WhisperModel | None = None

        logger.info(
            f"Initializing WhisperTranscriber (model={model_size}, device={self.device}, compute_type={self.compute_type})"
        )

    def _determine_device(self, device: DeviceType) -> str:
        """Determine the best device to use.

        Args:
            device: Requested device type

        Returns:
            str: Device to use (cuda or cpu)
        """
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU for transcription")
                return "cuda"
        except ImportError:
            pass

        logger.info("CUDA not available, using CPU for transcription")
        return "cpu"

    def _determine_compute_type(self, compute_type: ComputeType) -> str:
        """Determine the appropriate compute type based on device.

        Args:
            compute_type: Requested compute type

        Returns:
            str: Compute type to use
        """
        # If using CPU, force float32 or int8
        if self.device == "cpu" and compute_type in ("float16", "int8_float16"):
            logger.warning(f"Compute type {compute_type} not supported on CPU, using int8")
            return "int8"
        return compute_type

    def load_model(self) -> None:
        """Load the Whisper model."""
        if self.model is not None:
            logger.debug("Model already loaded")
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Model loaded successfully")
        except ImportError as e:
            logger.error("faster-whisper not installed. Run: uv add faster-whisper", exc_info=e)
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=e)
            raise

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Callable[[str], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            beam_size: Beam size for decoding
            vad_filter: Whether to use VAD filtering
            on_segment: Optional callback function called with each segment text as it's transcribed.
                       Signature: on_segment(text: str) -> None

        Returns:
            TranscriptionResult: Transcription result with text and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load model if not already loaded
        self.load_model()

        if self.model is None:
            raise RuntimeError("Failed to load Whisper model")

        logger.info(f"Transcribing audio file: {audio_path}")

        # Transcribe
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        # Collect segments and build full text
        segment_list = []
        full_text_parts = []

        for segment in segments:
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            segment_list.append(segment_dict)
            full_text_parts.append(segment.text)

            # Call callback if provided
            if on_segment:
                on_segment(segment.text)

        full_text = " ".join(full_text_parts).strip()

        logger.info(f"Transcription complete: {len(segment_list)} segments, language={info.language}")

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            segments=segment_list,
            duration=info.duration,
            device_used=self.device,
        )


def transcribe_audio(
    audio_path: Path | str,
    model_size: str = "base",
    language: str | None = None,
    device: DeviceType = "auto",
    compute_type: ComputeType = "float16",
    on_segment: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (None for auto-detection)
        device: Device to use (cuda, cpu, auto)
        compute_type: Compute type for model inference
        on_segment: Optional callback function called with each segment text as it's transcribed.
                   Signature: on_segment(text: str) -> None

    Returns:
        TranscriptionResult: Transcription result with text and metadata
    """
    transcriber = WhisperTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    return transcriber.transcribe(audio_path, language=language, on_segment=on_segment)
