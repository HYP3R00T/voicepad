# inference/model_manager.py

"""Whisper model loading, caching, and lifecycle management.

Models are loaded once and kept resident in GPU memory for the duration
of the application session. This avoids repeated multi-second load times.

Public API:
    load(model_name, device, compute_type)  -> WhisperModel
    unload(model_name)                      -> None
    unload_all()                            -> None
    is_loaded(model_name)                   -> bool
    get(model_name)                         -> WhisperModel | None
"""

from __future__ import annotations

import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from .constants import (
    COMPUTE_TYPE,
    CPU_COMPUTE_TYPE,
    CUDA_ERROR_KEYWORDS,
    DEFAULT_MODEL,
    DEVICE,
)
from .download import ensure_model_downloaded
from .errors import TranscriptionError
from ..config import get_config

logger = logging.getLogger(__name__)

# Module-level cache — keyed by (model_name, device, compute_type)
_model_cache: dict[tuple[str, str, str], WhisperModel] = {}

_session_logger: logging.Logger | None = None


def set_model_manager_session_logger(session_logger: logging.Logger | None) -> None:
    """Set the session logger for detailed model manager logging.

    Args:
        session_logger: Logger instance for the current transcription session
    """
    global _session_logger
    _session_logger = session_logger


def load(
    model_name: str = DEFAULT_MODEL,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
) -> WhisperModel:
    """Load a Whisper model and keep it resident in memory.

    If the model is already loaded (same name + device + compute_type),
    returns the cached instance immediately — no reload.

    If CUDA fails at load time, automatically falls back to CPU int8
    and logs a warning.

    Args:
        model_name:   Whisper model name (e.g. 'turbo').
        device:       'cuda' or 'cpu'.
        compute_type: CTranslate2 precision string.

    Returns:
        Loaded WhisperModel instance.

    Raises:
        TranscriptionError: If the model cannot be loaded on any device.
        ModelNotFoundError: If the model files are missing and download fails.
    """
    slog = _session_logger
    cache_key = (model_name, device, compute_type)

    if cache_key in _model_cache:
        logger.debug(f"Model cache hit: {cache_key}")
        if slog:
            slog.info(f"Model cache hit: {model_name} on {device} ({compute_type})")
            slog.debug(f"  Cache key: {cache_key}")
            slog.debug(f"  Model object: {type(_model_cache[cache_key]).__name__}")
        return _model_cache[cache_key]

    if slog:
        slog.info("Model not in cache, loading from disk...")
        slog.debug(f"  Model name: {model_name}")
        slog.debug(f"  Device: {device}")
        slog.debug(f"  Compute type: {compute_type}")
        slog.debug(f"  Cache key: {cache_key}")

    # Ensure weights are present before attempting to load
    if slog:
        slog.info("Checking if model is downloaded...")

    ensure_model_downloaded(model_name)

    if slog:
        slog.info("Model files confirmed present")

    logger.info(f"Loading '{model_name}' on {device} ({compute_type})")
    if slog:
        slog.info("Initializing WhisperModel...")

    load_start = time.perf_counter()

    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        load_ms = (time.perf_counter() - load_start) * 1000

        msg = f"Model '{model_name}' loaded in {load_ms:.0f}ms — cached as {cache_key}"
        logger.info(msg)

        if slog:
            slog.info(msg)
            slog.debug(f"  Model type: {type(model).__name__}")
            try:
                model_device = getattr(model, "device", "unknown")
                model_compute = getattr(model, "compute_type", "unknown")
                slog.debug(f"  Model device: {model_device}")
                slog.debug(f"  Model compute type: {model_compute}")
            except Exception:
                slog.debug("  Model device/compute_type: unavailable")

            # Try to get model size info
            try:
                from pathlib import Path

                model_path = Path.home() / ".cache" / "huggingface" / "hub"
                if model_path.exists():
                    slog.debug(f"  Model cache path: {model_path}")
            except Exception:
                pass

        _model_cache[cache_key] = model

        if slog:
            slog.info(f"Model added to cache (total cached: {len(_model_cache)})")

        # Run a short dummy inference to force CTranslate2 to allocate CUDA
        # compute buffers and warm up GPU kernels. Without this, the first real
        # transcription call pays the ~1-3s kernel initialization cost instead
        # of it happening here during model warm-up.
        _warmup_model(model, slog)

        return model

    except RuntimeError as e:
        if _is_cuda_error(e):
            msg = f"CUDA unavailable at load time: {e}\n  Falling back to CPU ({CPU_COMPUTE_TYPE})."
            logger.warning(msg)
            if slog:
                slog.warning(msg)
                slog.debug(f"  Original device: {device}")
                slog.debug(f"  Original compute type: {compute_type}")
                slog.debug("  Fallback device: cpu")
                slog.debug(f"  Fallback compute type: {CPU_COMPUTE_TYPE}")
            return _load_cpu_fallback(model_name)

        msg = f"Failed to load model '{model_name}': {e}"
        if slog:
            slog.error(msg)
        raise TranscriptionError(msg) from e

    except Exception as e:
        msg = f"Failed to load model '{model_name}': {e}"
        if slog:
            slog.error(msg)
            slog.debug(f"  Exception type: {type(e).__name__}")
        raise TranscriptionError(msg) from e


def unload(model_name: str = DEFAULT_MODEL) -> None:
    """Remove a model from the cache and free its memory.

    All cache entries matching model_name are removed, regardless of
    device or compute_type. Safe to call even if the model is not loaded.

    Args:
        model_name: Whisper model name to unload.
    """
    keys_to_remove = [k for k in _model_cache if k[0] == model_name]
    for key in keys_to_remove:
        del _model_cache[key]
        logger.info(f"Model unloaded: {key}")

    if not keys_to_remove:
        logger.debug(f"unload('{model_name}') called but model was not cached.")


def unload_all() -> None:
    """Unload every cached model and clear the cache entirely.

    Call this on TUI exit to cleanly free all GPU memory.
    """
    count = len(_model_cache)
    _model_cache.clear()
    logger.info(f"All models unloaded ({count} entries cleared).")


def is_loaded(model_name: str = DEFAULT_MODEL) -> bool:
    """Return True if any variant of model_name is currently cached.

    Args:
        model_name: Whisper model name to check.
    """
    return any(k[0] == model_name for k in _model_cache)


def get(
    model_name: str = DEFAULT_MODEL,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
) -> WhisperModel | None:
    """Return a cached model instance, or None if not loaded.

    Does not trigger a load. Use load() if you want automatic loading.

    Args:
        model_name:   Whisper model name.
        device:       Device the model was loaded on.
        compute_type: Precision the model was loaded with.

    Returns:
        WhisperModel if found in cache, None otherwise.
    """
    return _model_cache.get((model_name, device, compute_type))


def _is_cuda_error(e: Exception) -> bool:
    """Return True if the exception indicates a CUDA runtime failure."""
    return any(kw in str(e).lower() for kw in CUDA_ERROR_KEYWORDS)


def _warmup_model(model: WhisperModel, slog: logging.Logger | None) -> None:
    """Run a silent dummy inference to pre-allocate CUDA compute buffers.

    CTranslate2 defers GPU kernel initialization until the first actual
    inference call. Running a short silent pass here ensures that cost is
    paid during warm-up, not on the user's first real transcription.

    Args:
        model: The freshly loaded WhisperModel instance.
        slog:  Optional session logger.
    """
    config = get_config()
    if not config.model_warmup_enabled:
        return

    try:
        warmup_start = time.perf_counter()
        warmup_samples = max(1, int(config.model_warmup_duration_s * 16_000))
        dummy_audio = np.zeros(warmup_samples, dtype=np.float32)
        segs, _ = model.transcribe(
            dummy_audio,
            language=config.model_warmup_language,
            beam_size=config.model_warmup_beam_size,
            vad_filter=config.model_warmup_vad_filter,
        )
        list(segs)
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        msg = f"Model warm-up complete in {warmup_ms:.0f}ms"
        logger.info(msg)
        if slog:
            slog.info(msg)
    except Exception as e:
        # Warm-up failure is non-fatal — log and continue
        logger.warning(f"Model warm-up failed (non-fatal): {e}")
        if slog:
            slog.warning(f"Model warm-up failed (non-fatal): {e}")


def _load_cpu_fallback(model_name: str) -> WhisperModel:
    """Load the model on CPU with int8 precision.

    Used automatically when CUDA fails. Result is cached under the
    (model_name, 'cpu', 'int8') key.

    Args:
        model_name: Whisper model name.

    Returns:
        WhisperModel loaded on CPU.

    Raises:
        TranscriptionError: If CPU load also fails.
    """
    cache_key = (model_name, "cpu", CPU_COMPUTE_TYPE)

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        logger.info(f"Loading '{model_name}' on CPU ({CPU_COMPUTE_TYPE})")
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=CPU_COMPUTE_TYPE,
        )
        _model_cache[cache_key] = model
        return model

    except Exception as e:
        raise TranscriptionError(f"CPU fallback also failed for '{model_name}': {e}") from e
