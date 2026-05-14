"""Model loading and caching for Whisper models."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

from .constants import CUDA_ERROR_KEYWORDS
from .exceptions import TranscriptionError

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)

_model_cache: dict[tuple[str, str, str], WhisperModel] = {}


def _is_cuda_error(e: Exception) -> bool:
    """Check if exception indicates CUDA runtime failure."""
    return any(kw in str(e).lower() for kw in CUDA_ERROR_KEYWORDS)


def _load_cpu_fallback(model_name: str, download_root: str | None = None) -> tuple[WhisperModel, str, str]:
    """Load model on CPU with int8 precision.

    Args:
        model_name: Whisper model name
        download_root: Optional directory for model cache

    Returns:
        Tuple of (model, device, compute_type)
    """
    cache_key = (model_name, "cpu", "int8")
    if cache_key in _model_cache:
        return _model_cache[cache_key], "cpu", "int8"
    logger.info(f"Loading '{model_name}' on CPU (int8)")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=download_root)
    _model_cache[cache_key] = model
    return model, "cpu", "int8"


def get_or_load_model(config: Config) -> tuple[WhisperModel, str, str, bool]:
    """Load or retrieve cached Whisper model.

    Handles device selection (CUDA/CPU) and automatic fallback on GPU errors.
    Models are cached per (model_name, device, compute_type) combination.

    Args:
        config: Configuration with model settings

    Returns:
        Tuple of (model, device, compute_type, fallback_to_cpu)

    Raises:
        TranscriptionError: If model cannot be loaded
    """
    model_name = config.transcription_model

    cfg_device = getattr(config, "transcription_device", "auto")
    device = "cuda" if cfg_device == "auto" else cfg_device

    cfg_compute = getattr(config, "transcription_compute_type", "auto")
    compute = "int8" if cfg_compute == "auto" else cfg_compute

    fallback = False

    cache_key = (model_name, device, compute)
    if cache_key in _model_cache:
        logger.debug(f"Model cache hit: {cache_key}")
        return _model_cache[cache_key], device, compute, fallback

    logger.info(f"Loading '{model_name}' on {device} ({compute})")
    load_start = time.perf_counter()

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
