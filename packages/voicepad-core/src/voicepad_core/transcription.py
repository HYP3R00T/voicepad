"""Audio transcription using faster-whisper (CTranslate2 backend).

Primary entry point: transcribe_buffer(audio, config) -> TranscriptionResult
All transcription goes through this function — no file I/O required.
transcribe_file() is a thin convenience wrapper that loads audio then calls transcribe_buffer().

GPU support:
    CUDA DLLs are provided by nvidia-cublas-cu12 and nvidia-cudnn-cu12 — no system
    CUDA installation required. CTranslate2 finds them automatically via the Python
    package path. If CUDA is unavailable for any reason, falls back to CPU transparently.

Accuracy:
    Uses standard model.transcribe with vad_filter=True for all durations.
    VAD splits audio at natural speech pauses, avoiding the hallucinated ellipses
    that BatchedInferencePipeline produces at fixed 30s chunk boundaries.
    On turbo/RTX 3050, standard+VAD is also faster than batched mode.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
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
# if CUDA is unavailable. CUDA DLLs are provided by nvidia-cublas-cu12 and
# nvidia-cudnn-cu12 — no system CUDA installation required.
# Overridden at runtime by config.transcription_device.
DEVICE: str = "cuda"

# Compute precision for CTranslate2.
# int8 gives the best speed/VRAM trade-off on an RTX 3050 (4 GB).
# Overridden at runtime by config.transcription_compute_type.
COMPUTE_TYPE: str = "int8"

# Beam search width.
# 3 = balanced. 1 is fastest but more likely to repeat or drift on longer clips.
# 5 = most accurate, ~25% slower.
BEAM_SIZE: int = 3

# Language passed to Whisper.
# Fixed to "en" — skips language detection (~300ms saved per call).
# Change to None to re-enable auto-detection for multilingual use.
LANGUAGE: str = "en"

# Suppress hallucinated segments at silence boundaries.
HALLUCINATION_SILENCE_THRESHOLD: float = 0.5

# Suppress segments where speech probability is below this threshold.
# Higher = more aggressive suppression of noise/silence hallucinations.
# Default faster-whisper value is 0.6; 0.8 better suppresses tail noise.
NO_SPEECH_THRESHOLD: float = 0.8

# Initial prompt to prime Whisper toward punctuated, well-formatted output.
# This is a soft hint — Whisper may still omit punctuation on very short clips.
INITIAL_PROMPT: str = "Hello. This is a transcription with proper punctuation, capitalization, and grammar."

# Models that use the true distil-whisper architecture (reduced decoder layers).
# These work best with condition_on_previous_text=False per the faster-whisper docs.
# Note: turbo and large-v3-turbo are full large-v3 fine-tunes — NOT distil-whisper.
# They support initial_prompt and condition_on_previous_text normally.
_DISTIL_MODELS: frozenset[str] = frozenset({
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
})

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
# Model cache — one WhisperModel per (model, device, compute_type) combination
# ---------------------------------------------------------------------------

_model_cache: dict[tuple[str, str, str], WhisperModel] = {}


def _is_cuda_error(e: Exception) -> bool:
    """Return True if the exception looks like a CUDA runtime failure."""
    return any(kw in str(e).lower() for kw in _CUDA_ERROR_KEYWORDS)


def _trim_trailing_silence(
    audio: np.ndarray, sr: int = 16000, threshold: float = 0.005, window_s: float = 0.3
) -> np.ndarray:
    """Remove trailing silence/noise from audio.

    Whisper hallucinates on trailing noise — it creates phantom segments that
    extend beyond the actual speech and fills them with invented text.
    Trimming the tail prevents this entirely.

    Args:
        threshold: RMS below this is considered silence (default 0.005 ≈ -46dBFS).
        window_s:  Step size for the scan (default 0.3s).
    """
    window = int(window_s * sr)
    if len(audio) <= window:
        return audio
    end = len(audio)
    while end > window:
        rms = float(np.sqrt(np.mean(audio[end - window : end] ** 2)))
        if rms > threshold:
            # Add a small pad so we don't cut off the very end of speech
            end = min(len(audio), end + window)
            break
        end -= window // 2
    return audio[:end]


def _load_cpu_fallback(model_name: str, download_root: str | None = None) -> tuple[WhisperModel, str, str]:
    """Load model on CPU and cache it. Returns (model, 'cpu', 'int8')."""
    cache_key = (model_name, "cpu", "int8")
    if cache_key in _model_cache:
        return _model_cache[cache_key], "cpu", "int8"
    logger.info(f"Loading '{model_name}' on CPU (int8)")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=download_root)
    _model_cache[cache_key] = model
    return model, "cpu", "int8"


# ---------------------------------------------------------------------------


# Model download helpers
# ---------------------------------------------------------------------------


def _get_repo_id(model_name: str) -> str:
    """Resolve the HuggingFace repo ID for a model name.

    Uses faster-whisper's own _MODELS dict so non-Systran models
    (turbo, large-v3-turbo, distil-large-v3.5) resolve correctly.
    Falls back to the Systran prefix for unknown names.
    """
    try:
        from faster_whisper.utils import _MODELS  # type: ignore[attr-defined]

        if model_name in _MODELS:
            return _MODELS[model_name]
    except Exception:
        pass
    return f"{_HF_REPO_PREFIX}{model_name}"


def model_downloaded(model_name: str, config: Config | None = None) -> bool:
    """Return True if the model weights (model.bin) are in the local cache.

    Checks for the actual weight file, not just metadata.
    Does not make any network requests.
    """
    import os
    from pathlib import Path as _Path

    # Use config's model_cache_path if provided, otherwise fall back to HF default
    if config is not None:
        cache_root = config.model_cache_path / "hub"
    else:
        hf_home = os.environ.get("HF_HOME", "")
        cache_root = _Path(hf_home) / "hub" if hf_home else _Path.home() / ".cache" / "huggingface" / "hub"

    repo_id = _get_repo_id(model_name)
    snapshots = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"

    if not snapshots.exists():
        return False

    # model.bin is the CTranslate2 weight file — its presence means the
    # download completed. Config/tokenizer files alone are not enough.
    return any(snap.is_dir() and (snap / "model.bin").exists() for snap in snapshots.iterdir())


def ensure_model_downloaded(
    model_name: str,
    config: Config | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Download the model weights if not already cached. Blocks until complete.

    Args:
        on_progress: Optional callback(downloaded_bytes, total_bytes) called
                     during download. total_bytes may be 0 if unknown.

    Raises:
        TranscriptionError: If the download fails.
    """
    if model_downloaded(model_name, config):
        return

    repo_id = _get_repo_id(model_name)
    logger.info(f"Downloading '{model_name}' from {repo_id}")

    cache_dir: str | None = None
    if config is not None:
        hub_dir = config.model_cache_path / "hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(hub_dir)

    tqdm_class = None
    if on_progress is not None:
        _cb = on_progress
        from tqdm.auto import tqdm as _tqdm

        # snapshot_download creates one tqdm instance per file.
        # model.bin is the only large file (~99% of the download).
        # We identify it by its size (> 1 MB) and track only its bytes.
        # The total comes from the tqdm instance itself (set from the HTTP
        # Content-Length header by huggingface_hub) — no extra API call needed.
        _downloaded = [0]
        _total = [0]

        class _ProgressTqdm(_tqdm):  # ty:ignore[unsupported-base]
            def __init__(self, *args: object, **kwargs: object) -> None:
                kwargs.setdefault("disable", True)
                super().__init__(*args, **kwargs)
                # Only track this instance if it looks like model.bin (> 1 MB).
                # Small JSON/config files are ignored so they don't pollute the
                # progress or cause a premature 100%.
                self._track = bool(self.total and int(self.total) > 1_000_000)
                if self._track and _total[0] == 0 and self.total:
                    _total[0] = int(self.total)

            def update(self, n: int = 1) -> bool | None:
                result = super().update(n)
                if self._track and n and n > 0:
                    _downloaded[0] += n
                    _cb(_downloaded[0], _total[0])
                return result

        tqdm_class = _ProgressTqdm

    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
            tqdm_class=tqdm_class,
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

    Respects config.transcription_device and config.transcription_compute_type.
    Falls back to CPU on any CUDA-related error.

    Returns:
        (model, actual_device, actual_compute_type, fallback_occurred)

    Raises:
        TranscriptionError: If the model cannot be loaded on any device.
    """
    model_name = config.transcription_model

    # Resolve device from config
    cfg_device = getattr(config, "transcription_device", "auto")
    device = "cuda" if cfg_device == "auto" else cfg_device

    # Resolve compute type from config
    cfg_compute = getattr(config, "transcription_compute_type", "auto")
    compute = "int8" if cfg_compute == "auto" else cfg_compute

    fallback = False

    cache_key = (model_name, device, compute)
    if cache_key in _model_cache:
        logger.debug(f"Model cache hit: {cache_key}")
        return _model_cache[cache_key], device, compute, fallback

    logger.info(f"Loading '{model_name}' on {device} ({compute})")
    load_start = time.perf_counter()

    # Resolve model download root from config
    download_root: str | None = None
    if hasattr(config, "model_cache_path"):
        model_dir = config.model_cache_path / "hub"
        model_dir.mkdir(parents=True, exist_ok=True)
        download_root = str(model_dir)

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute, download_root=download_root)
        load_ms = (time.perf_counter() - load_start) * 1000
        logger.info(f"Model loaded in {load_ms:.0f}ms — cached as {cache_key}")
        _model_cache[cache_key] = model
        return model, device, compute, fallback

    except RuntimeError as e:
        if _is_cuda_error(e):
            logger.warning(f"CUDA unavailable at load time: {e} — falling back to CPU")
            model, device, compute = _load_cpu_fallback(model_name, download_root)
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

    # Trim trailing silence/noise to prevent Whisper hallucinating on the tail
    audio = _trim_trailing_silence(audio)

    duration_s = len(audio) / 16000

    if duration_s < MIN_AUDIO_DURATION_S:
        raise AudioTooShortError(f"Audio is {duration_s:.2f}s — below minimum {MIN_AUDIO_DURATION_S}s")

    if duration_s > MAX_AUDIO_DURATION_S:
        logger.info(f"Long recording: {duration_s:.1f}s — transcription will take a moment")

    model, device, compute, fallback = get_or_load_model(config)

    # Distil models do not benefit from previous-text conditioning or prompts.
    # For non-distil models, we enable conditioning for better punctuation BUT
    # use aggressive VAD chunking (shorter max_speech_duration) to prevent
    # hallucination buildup across long audio.
    is_distil = config.transcription_model in _DISTIL_MODELS
    condition_on_prev = not is_distil
    prompt = None if is_distil else INITIAL_PROMPT

    try:
        # Standard mode with VAD — splits at natural speech pauses.
        # Avoids hallucinated ellucinations that BatchedInferencePipeline produces
        # at fixed 30s chunk boundaries. On turbo/RTX 3050, also faster than batched.
        # VAD parameters tuned to prevent hallucinations while maintaining punctuation:
        # - max_speech_duration_s: 30s chunks prevent error buildup with conditioning
        # - min_silence_duration_ms: 500ms for more frequent natural breaks
        # - speech_pad_ms: 400ms to avoid cutting words at boundaries
        vad_params = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": 30.0,  # Force split at 30s to prevent hallucinations
            "min_silence_duration_ms": 500,  # 0.5s silence for more natural breaks
            "speech_pad_ms": 400,
        }
        segments_iter, info = model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters=vad_params,
            hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            initial_prompt=prompt,
            condition_on_previous_text=condition_on_prev,
        )
        segments = [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments_iter]

    except RuntimeError as e:
        if _is_cuda_error(e):
            # CUDA failed during inference — evict GPU entry, retry on CPU.
            logger.warning(f"CUDA inference error: {e} — retrying on CPU")
            _model_cache.pop((config.transcription_model, device, compute), None)
            download_root_fb: str | None = None
            if hasattr(config, "model_cache_path"):
                download_root_fb = str(config.model_cache_path / "hub")
            cpu_model, device, compute = _load_cpu_fallback(config.transcription_model, download_root_fb)
            fallback = True
            segments_iter, info = cpu_model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters=vad_params,
                hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                initial_prompt=prompt,
                condition_on_previous_text=condition_on_prev,
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
