"""Core transcription functionality using faster-whisper."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "cpu", "auto"]
ComputeType = Literal["int8", "int8_float16", "int16", "float16", "float32", "auto"]


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
        device: DeviceType = "auto",
        compute_type: ComputeType = "auto",
        language: str | None = None,
        poll_interval: float = 30.0,
        min_duration: float = 5.0,
    ) -> None:
        """Initialize the poller.

        Args:
            model_size: Whisper model size
            device: Device to use (cuda, cpu, auto)
            compute_type: Compute type for model inference
            language: Language code (None for auto-detection)
            poll_interval: Seconds between transcription polls
            min_duration: Minimum audio duration in seconds before first transcription attempt
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.poll_interval = poll_interval
        self.min_duration = min_duration
        self.transcriber = WhisperTranscriber(model_size, device, compute_type)
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

        # Check file size; need enough audio
        if not audio_path.exists():
            return None

        file_size = audio_path.stat().st_size
        # Rough estimate: 44100 Hz, float32 = ~176KB per second + WAV header
        estimated_duration = max(0.0, (file_size - 1000) / (44100 * 4))

        if estimated_duration < self.min_duration:
            return None

        try:
            # Transcribe the file
            result = self.transcriber.transcribe(
                audio_path,
                language=self.language,
                on_segment=None,  # Handle deduplication ourselves
            )

            # Filter to only new segments (those after last_segment_end)
            new_segments = [seg for seg in result.segments if seg["end"] > self.last_segment_end]

            with self._lock:
                if new_segments:
                    self.last_segment_end = result.segments[-1]["end"]
                    # Call callback for each new segment
                    for seg in new_segments:
                        if on_segment:
                            on_segment(seg["text"])

            # Return full result with all segments (not just new)
            return result

        except Exception as e:
            logger.warning(f"Transcription poll failed: {e}")
            return None


class WhisperTranscriber:
    """Wrapper for faster-whisper transcription with GPU support."""

    def __init__(
        self,
        model_size: str = "base",
        device: DeviceType = "auto",
        compute_type: ComputeType = "auto",
    ) -> None:
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (cuda, cpu, auto)
            compute_type: Compute type for model inference (or auto to choose best option)
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
        if compute_type == "auto":
            resolved = "float16" if self.device == "cuda" else "int8"
            logger.info(f"Auto compute type selected: {resolved} (device={self.device})")
            return resolved

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
    compute_type: ComputeType = "auto",
    on_segment: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (None for auto-detection)
        device: Device to use (cuda, cpu, auto)
        compute_type: Compute type for model inference (or auto to choose best option)
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
