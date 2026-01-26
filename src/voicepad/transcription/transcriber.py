"""Core transcription functionality using faster-whisper."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: str
    segments: list[dict]
    duration: float
    device_used: str


class TranscriptionPoller:
    """Poll a growing audio file and transcribe it incrementally."""

    def __init__(
        self,
        model_size: str = "base",
        poll_interval: float = 30.0,
        min_duration: float = 5.0,
    ) -> None:
        """Initialize the poller.

        Args:
            model_size: Whisper model size
            poll_interval: Seconds between transcription polls
            min_duration: Minimum audio duration in seconds before first transcription attempt
        """
        self.model_size = model_size
        self.poll_interval = poll_interval
        self.min_duration = min_duration
        self.transcriber = WhisperTranscriber(model_size)
        self.last_segment_end: float = 0.0
        self._lock = threading.Lock()

    def poll_and_transcribe(
        self,
        audio_file: Path | str,
        on_segment: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> TranscriptionResult | None:
        """Poll file and transcribe accumulated audio.

        Args:
            audio_file: Path to growing audio file
            on_segment: Callback for each new segment
            stop_event: Threading event to check if recording is still active

        Returns:
            TranscriptionResult if transcription succeeds, None if file too short
        """
        audio_path = Path(audio_file)

        # Check file size; need enough audio (rough estimate: 44100 Hz, float32)
        if not audio_path.exists():
            return None

        if (audio_path.stat().st_size - 1000) / (44100 * 4) < self.min_duration:
            return None

        try:
            # Transcribe the file
            result = self.transcriber.transcribe(
                audio_path,
                on_segment=None,  # Handle deduplication ourselves
            )

            new_segments = self._filter_new_segments(result.segments)

            with self._lock:
                if new_segments:
                    self.last_segment_end = result.segments[-1]["end"]
                    if on_segment:
                        for seg in new_segments:
                            on_segment(seg["text"])

            # Return full result with all segments (not just new)
            return result

        except Exception as e:
            logger.warning(f"Transcription poll failed: {e}")
            return None

    def _filter_new_segments(self, segments: list[dict]) -> list[dict]:
        """Return only segments that start after the last processed end time."""
        return [seg for seg in segments if seg["end"] > self.last_segment_end]


class WhisperTranscriber:
    """Wrapper for faster-whisper transcription with GPU support."""

    def __init__(
        self,
        model_size: str = "base",
    ) -> None:
        """Initialize the transcriber."""
        self.model_size = model_size
        self.device, self.compute_type = self._resolve_device_and_compute()
        self.model: WhisperModel | None = None

        logger.info(
            f"Initializing WhisperTranscriber (model={model_size}, device={self.device}, compute_type={self.compute_type})"
        )

    def _resolve_device_and_compute(self) -> tuple[str, str]:
        """Determine device and compute type based on system capabilities."""
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

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
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Callable[[str], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
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
    on_segment: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        on_segment: Optional callback function called with each segment text as it's transcribed.
                   Signature: on_segment(text: str) -> None

    Returns:
        TranscriptionResult: Transcription result with text and metadata
    """
    transcriber = WhisperTranscriber(model_size=model_size)
    return transcriber.transcribe(audio_path, on_segment=on_segment)
