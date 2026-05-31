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

from faster_whisper import WhisperModel

from .constants import (
    COMPUTE_TYPE,
    CPU_COMPUTE_TYPE,
    CUDA_ERROR_KEYWORDS,
    DEFAULT_MODEL,
    DEVICE,
)
from .download import ensure_model_downloaded
from .exceptions import TranscriptionError

logger = logging.getLogger(__name__)

# Module-level cache — keyed by (model_name, device, compute_type)
_model_cache: dict[tuple[str, str, str], WhisperModel] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    cache_key = (model_name, device, compute_type)

    if cache_key in _model_cache:
        logger.debug(f"Model cache hit: {cache_key}")
        return _model_cache[cache_key]

    # Ensure weights are present before attempting to load
    ensure_model_downloaded(model_name)

    logger.info(f"Loading '{model_name}' on {device} ({compute_type})")
    load_start = time.perf_counter()

    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        load_ms = (time.perf_counter() - load_start) * 1000
        logger.info(f"Model '{model_name}' loaded in {load_ms:.0f}ms — cached as {cache_key}")
        _model_cache[cache_key] = model
        return model

    except RuntimeError as e:
        if _is_cuda_error(e):
            logger.warning(f"CUDA unavailable at load time: {e}\n  Falling back to CPU ({CPU_COMPUTE_TYPE}).")
            return _load_cpu_fallback(model_name)
        raise TranscriptionError(f"Failed to load model '{model_name}': {e}") from e

    except Exception as e:
        raise TranscriptionError(f"Failed to load model '{model_name}': {e}") from e


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_cuda_error(e: Exception) -> bool:
    """Return True if the exception indicates a CUDA runtime failure."""
    return any(kw in str(e).lower() for kw in CUDA_ERROR_KEYWORDS)


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
