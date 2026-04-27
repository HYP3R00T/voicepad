"""Audio transcription using faster-whisper (CTranslate2 backend).

Primary entry point: transcribe_buffer(audio, config) -> TranscriptionResult
All transcription goes through this function — no file I/O required.
transcribe_file() is a thin convenience wrapper that loads audio then calls transcribe_buffer().
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transcription constants — internal tuning, not user-facing config
# ---------------------------------------------------------------------------

# Target device. "cuda" is tried first; falls back to "cpu" automatically
# if CUDA libraries are absent or a runtime error occurs at model load.
DEVICE: str = "cuda"

# Compute precision for CTranslate2.
# int8 gives the best speed/VRAM trade-off on an RTX 3050 (4 GB).
# Switch to float16 if you want slightly higher accuracy at the cost of ~2x VRAM.
COMPUTE_TYPE: str = "int8"

# Beam search width: 3 is a good balance of speed and accuracy.
# Increase to 5 for better accuracy at the cost of ~30% more latency.
BEAM_SIZE: int = 3

# Language passed to Whisper. None = auto-detect per utterance.
# Set to "en" to skip language detection and save ~200ms per call.
LANGUAGE: str | None = None

# Minimum audio duration to attempt transcription.
# Recordings shorter than this are almost certainly silence or noise.
MIN_AUDIO_DURATION_S: float = 0.5

# Soft upper bound. Transcription still runs but a warning is logged.
MAX_AUDIO_DURATION_S: float = 30.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A single transcription segment with timestamps."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of a transcription call.

    Attributes:
        text:                 Full transcription text (all segments joined).
        segments:             Individual timed segments.
        language:             Detected or configured language code (e.g. "en").
        language_probability: Confidence of language detection (0.0–1.0).
        duration_s:           Audio duration in seconds.
        latency_ms:           Wall-clock time from call entry to result (ms).
        device:               Actual device used ("cuda" or "cpu").
        compute_type:         Actual compute type used (e.g. "int8", "float16").
        fallback_to_cpu:      True if CUDA was requested but fell back to CPU.
    """

    text: str
    segments: list[Segment]
    language: str
    language_probability: float
    duration_s: float
    latency_ms: float
    device: str
    compute_type: str
    fallback_to_cpu: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TranscriptionError(Exception):
    """Raised when transcription cannot be completed."""


class AudioTooShortError(TranscriptionError):
    """Raised when audio is shorter than MIN_AUDIO_DURATION_S."""


class AudioTooLongWarning(UserWarning):
    """Issued when audio exceeds MAX_AUDIO_DURATION_S."""


# ---------------------------------------------------------------------------
# Model cache — one instance per (model, device, compute_type) combination
# ---------------------------------------------------------------------------

_model_cache: dict[tuple[str, str, str], WhisperModel] = {}


def _cuda_libraries_available() -> bool:
    """Return True if the NVIDIA CUDA Python packages are importable."""
    try:
        importlib.import_module("nvidia.cublas")
        importlib.import_module("nvidia.cudnn")
        return True
    except ImportError:
        return False


def get_or_load_model(config: Config) -> tuple[WhisperModel, str, str, bool]:
    """Return a cached WhisperModel, loading it if necessary.

    Handles CUDA → CPU fallback transparently when CUDA libraries are absent
    or a CUDA runtime error occurs during model load.

    Args:
        config: Configuration with model name.

    Returns:
        Tuple of (model, actual_device, actual_compute_type, fallback_occurred).

    Raises:
        TranscriptionError: If the model cannot be loaded on any device.
    """
    device = DEVICE
    compute = COMPUTE_TYPE
    model_name = config.transcription_model
    fallback = False

    if device == "cuda" and not _cuda_libraries_available():
        logger.warning("CUDA libraries not found — falling back to CPU")
        device, compute, fallback = "cpu", "int8", True

    cache_key = (model_name, device, compute)
    if cache_key in _model_cache:
        logger.debug(f"Model cache hit: {cache_key}")
        return _model_cache[cache_key], device, compute, fallback

    logger.info(f"Loading model '{model_name}' on {device} ({compute})")
    load_start = time.perf_counter()

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except RuntimeError as e:
        cuda_keywords = ("cublas", "cuda", "cudnn", "nvrtc")
        if device == "cuda" and any(kw in str(e).lower() for kw in cuda_keywords):
            logger.warning(f"CUDA runtime error: {e} — falling back to CPU")
            device, compute, fallback = "cpu", "int8", True
            cache_key = (model_name, device, compute)
            if cache_key in _model_cache:
                return _model_cache[cache_key], device, compute, fallback
            model = WhisperModel(model_name, device=device, compute_type=compute)
        else:
            raise TranscriptionError(f"Failed to load model '{model_name}': {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Failed to load model '{model_name}': {e}") from e

    load_ms = (time.perf_counter() - load_start) * 1000
    logger.info(f"Model loaded in {load_ms:.0f}ms — cached as {cache_key}")
    _model_cache[cache_key] = model
    return model, device, compute, fallback


# ---------------------------------------------------------------------------
# Core transcription functions
# ---------------------------------------------------------------------------


def transcribe_buffer(audio: np.ndarray, config: Config) -> TranscriptionResult:
    """Transcribe audio from a numpy array. No file I/O.

    Args:
        audio:  float32 numpy array at 16 kHz mono.
        config: Configuration (model, device, compute type).

    Returns:
        TranscriptionResult with text, segments, timing, and device info.

    Raises:
        AudioTooShortError: If audio is shorter than MIN_AUDIO_DURATION_S.
        TranscriptionError: If transcription fails.
    """
    import warnings

    call_start = time.perf_counter()

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()

    duration_s = len(audio) / 16000

    if duration_s < MIN_AUDIO_DURATION_S:
        raise AudioTooShortError(f"Audio is {duration_s:.2f}s — below minimum {MIN_AUDIO_DURATION_S}s")

    if duration_s > MAX_AUDIO_DURATION_S:
        warnings.warn(
            f"Audio is {duration_s:.1f}s — exceeds {MAX_AUDIO_DURATION_S}s, transcription may be slow",
            AudioTooLongWarning,
            stacklevel=2,
        )

    try:
        model, device, compute, fallback = get_or_load_model(config)

        segments_iter, info = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=False,
        )

        segments = [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter]
        text = " ".join(s.text for s in segments if s.text).strip()
        latency_ms = (time.perf_counter() - call_start) * 1000

        logger.info(
            f"Transcribed {duration_s:.1f}s in {latency_ms:.0f}ms on {device} ({compute}) — {len(segments)} segments"
        )

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration_s=duration_s,
            latency_ms=latency_ms,
            device=device,
            compute_type=compute,
            fallback_to_cpu=fallback,
        )

    except (AudioTooShortError, TranscriptionError):
        raise
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e


def transcribe_file(audio_path: Path, config: Config) -> TranscriptionResult:
    """Transcribe an audio file by loading it and calling transcribe_buffer().

    Args:
        audio_path: Path to a WAV (or any soundfile-readable) audio file.
        config:     Configuration object.

    Returns:
        TranscriptionResult — same as transcribe_buffer().

    Raises:
        TranscriptionError:  If the file cannot be read or transcription fails.
        AudioTooShortError:  If audio is shorter than MIN_AUDIO_DURATION_S.
    """
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise TranscriptionError(f"Path is not a file: {audio_path}")

    try:
        audio, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as e:
        raise TranscriptionError(f"Failed to read '{audio_path}': {e}") from e

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if file_sr != 16000:
        logger.warning(
            f"File sample rate is {file_sr} Hz, expected 16000 Hz — quality may be reduced (no resampling performed)"
        )

    return transcribe_buffer(audio, config)
