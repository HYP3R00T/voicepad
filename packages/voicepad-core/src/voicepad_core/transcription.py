"""Audio transcription using faster-whisper (CTranslate2 backend).

Primary entry point: transcribe_buffer(audio, config) -> TranscriptionResult
All transcription goes through this function — no file I/O required.
transcribe_file() is a thin convenience wrapper that loads audio then calls transcribe_buffer().

GPU support:
    CUDA DLLs are bundled with the torch package — no system CUDA installation required.
    faster-whisper (CTranslate2) finds them automatically when torch is installed.
    If CUDA is unavailable for any reason, falls back to CPU transparently.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transcription constants — internal tuning, not user-facing config
# ---------------------------------------------------------------------------

# Target device. "cuda" is tried first; falls back to "cpu" automatically
# if CUDA is unavailable. CUDA DLLs come bundled with torch — no system
# CUDA installation required.
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

# Initial prompt to prime Whisper toward punctuated, well-formatted output.
# This is a soft hint — Whisper may still omit punctuation on very short clips.
INITIAL_PROMPT: str = "Hello. This is a transcription with proper punctuation, capitalization, and grammar."

# Minimum audio duration to attempt transcription.
# Recordings shorter than this are almost certainly silence or noise.
MIN_AUDIO_DURATION_S: float = 0.5

# Soft upper bound — just log, no warning raised to the caller.
# Long-form dictation (10–15 min) is valid. This is only a performance note.
MAX_AUDIO_DURATION_S: float = 900.0

# Hugging Face repo prefix for faster-whisper models
_HF_REPO_PREFIX = "Systran/faster-whisper-"

# CUDA runtime error keywords — any of these in a RuntimeError means GPU failed
_CUDA_ERROR_KEYWORDS = ("cublas", "cuda", "cudnn", "nvrtc", "cufft", "curand")


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


def _is_cuda_error(e: Exception) -> bool:
    """Return True if the exception looks like a CUDA runtime failure."""
    return any(kw in str(e).lower() for kw in _CUDA_ERROR_KEYWORDS)


def _load_cpu_fallback(model_name: str) -> tuple[WhisperModel, str, str]:
    """Load model on CPU and cache it. Returns (model, 'cpu', 'int8')."""
    cache_key = (model_name, "cpu", "int8")
    if cache_key in _model_cache:
        return _model_cache[cache_key], "cpu", "int8"
    logger.info(f"Loading '{model_name}' on CPU (int8)")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    _model_cache[cache_key] = model
    return model, "cpu", "int8"


# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------


def model_downloaded(model_name: str) -> bool:
    """Return True if the model weights (model.bin) are in the local cache.

    Checks for the actual weight file, not just metadata.
    Does not make any network requests.
    """
    import os
    from pathlib import Path as _Path

    repo_id = f"{_HF_REPO_PREFIX}{model_name}"
    hf_home = os.environ.get("HF_HOME", "")
    cache_root = _Path(hf_home) if hf_home else _Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"

    if not snapshots.exists():
        return False

    # model.bin is the CTranslate2 weight file — its presence means the
    # download completed. Config/tokenizer files alone are not enough.
    return any(snap.is_dir() and (snap / "model.bin").exists() for snap in snapshots.iterdir())


def ensure_model_downloaded(model_name: str) -> None:
    """Download the model weights if not already cached. Blocks until complete.

    Raises:
        TranscriptionError: If the download fails.
    """
    if model_downloaded(model_name):
        return

    repo_id = f"{_HF_REPO_PREFIX}{model_name}"
    logger.info(f"Downloading '{model_name}' from {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            # Skip framework-specific weights we don't need
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
    except HfHubHTTPError as e:
        raise TranscriptionError(f"Failed to download model '{model_name}': {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Failed to download model '{model_name}': {e}") from e


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def get_or_load_model(config: Config) -> tuple[WhisperModel, str, str, bool]:
    """Return a cached WhisperModel, loading it if necessary.

    Attempts CUDA first. Falls back to CPU on any CUDA-related error.
    CUDA DLLs are provided by the bundled torch package — no system
    CUDA installation required.

    Returns:
        (model, actual_device, actual_compute_type, fallback_occurred)

    Raises:
        TranscriptionError: If the model cannot be loaded on any device.
    """
    model_name = config.transcription_model
    device = DEVICE
    compute = COMPUTE_TYPE
    fallback = False

    cache_key = (model_name, device, compute)
    if cache_key in _model_cache:
        logger.debug(f"Model cache hit: {cache_key}")
        return _model_cache[cache_key], device, compute, fallback

    logger.info(f"Loading '{model_name}' on {device} ({compute})")
    load_start = time.perf_counter()

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
        load_ms = (time.perf_counter() - load_start) * 1000
        logger.info(f"Model loaded in {load_ms:.0f}ms — cached as {cache_key}")
        _model_cache[cache_key] = model
        return model, device, compute, fallback

    except RuntimeError as e:
        if _is_cuda_error(e):
            logger.warning(f"CUDA unavailable at load time: {e} — falling back to CPU")
            model, device, compute = _load_cpu_fallback(model_name)
            return model, device, compute, True
        raise TranscriptionError(f"Failed to load model '{model_name}': {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Failed to load model '{model_name}': {e}") from e


# ---------------------------------------------------------------------------
# Core transcription functions
# ---------------------------------------------------------------------------


def transcribe_buffer(audio: np.ndarray, config: Config) -> TranscriptionResult:
    """Transcribe audio from a numpy array. No file I/O.

    Args:
        audio:  float32 numpy array at 16 kHz mono.
        config: Configuration (model name).

    Returns:
        TranscriptionResult with text, segments, timing, and device info.

    Raises:
        AudioTooShortError: If audio is shorter than MIN_AUDIO_DURATION_S.
        TranscriptionError: If transcription fails.
    """

    call_start = time.perf_counter()

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()

    duration_s = len(audio) / 16000

    if duration_s < MIN_AUDIO_DURATION_S:
        raise AudioTooShortError(f"Audio is {duration_s:.2f}s — below minimum {MIN_AUDIO_DURATION_S}s")

    if duration_s > MAX_AUDIO_DURATION_S:
        logger.info(f"Long recording: {duration_s:.1f}s — transcription will take a moment")

    model, device, compute, fallback = get_or_load_model(config)

    try:
        segments_iter, info = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=False,
            initial_prompt=INITIAL_PROMPT,
            condition_on_previous_text=True,
        )
        segments = [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter]

    except RuntimeError as e:
        if _is_cuda_error(e):
            # CUDA failed during inference (e.g. DLL found but GPU OOM or driver issue).
            # Evict the bad GPU entry, retry on CPU.
            logger.warning(f"CUDA inference error: {e} — retrying on CPU")
            _model_cache.pop((config.transcription_model, device, compute), None)
            cpu_model, device, compute = _load_cpu_fallback(config.transcription_model)
            fallback = True
            segments_iter, info = cpu_model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=False,
                initial_prompt=INITIAL_PROMPT,
                condition_on_previous_text=True,
            )
            segments = [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter]
        else:
            raise TranscriptionError(f"Transcription failed: {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

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


def transcribe_file(audio_path: Path, config: Config) -> TranscriptionResult:
    """Transcribe an audio file by loading it and calling transcribe_buffer().

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
