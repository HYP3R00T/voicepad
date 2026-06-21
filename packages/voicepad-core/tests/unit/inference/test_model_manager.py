"""Tests for voicepad_core.inference.model_manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from voicepad_core.inference.errors import TranscriptionError
from voicepad_core.inference.model_manager import (
    _is_cuda_error,
    _load_cpu_fallback,
    _model_cache,
    get,
    is_loaded,
    load,
    unload,
    unload_all,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear model cache before each test."""
    _model_cache.clear()
    yield
    _model_cache.clear()


def test_is_cuda_error_detects_cuda_keywords() -> None:
    """_is_cuda_error returns True for CUDA-related errors."""
    assert _is_cuda_error(RuntimeError("cublas error"))
    assert _is_cuda_error(RuntimeError("CUDA out of memory"))
    assert _is_cuda_error(RuntimeError("cudnn initialization failed"))
    assert _is_cuda_error(RuntimeError("nvrtc error"))


def test_is_cuda_error_returns_false_for_other_errors() -> None:
    """_is_cuda_error returns False for non-CUDA errors."""
    assert not _is_cuda_error(RuntimeError("generic error"))
    assert not _is_cuda_error(ValueError("invalid value"))
    assert not _is_cuda_error(Exception("something else"))


def test_is_cuda_error_case_insensitive() -> None:
    """_is_cuda_error is case-insensitive."""
    assert _is_cuda_error(RuntimeError("CUDA Error"))
    assert _is_cuda_error(RuntimeError("CuBLAS failed"))


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_caches_model(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """load() caches model after first load."""
    mock_model = Mock()
    mock_whisper.return_value = mock_model

    result1 = load("turbo", "cuda", "int8")
    result2 = load("turbo", "cuda", "int8")

    assert mock_whisper.call_count == 1
    assert result1 is result2


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_different_configs_creates_separate_cache_entries(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """load() creates separate cache entries for different configurations."""
    mock_whisper.return_value = Mock()

    load("turbo", "cuda", "int8")
    load("turbo", "cpu", "int8")
    load("base", "cuda", "int8")

    assert len(_model_cache) == 3


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_ensures_model_downloaded(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """load() calls ensure_model_downloaded before loading."""
    mock_whisper.return_value = Mock()

    load("turbo")

    mock_ensure.assert_called_once_with("turbo")


@patch(
    "voicepad_core.inference.model_manager.ensure_model_downloaded",
    side_effect=[Path("C:/models/turbo"), Path("C:/models/turbo")],
)
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_falls_back_to_cpu_on_cuda_error(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """load() falls back to CPU when CUDA fails."""
    mock_whisper.side_effect = [RuntimeError("cuda error"), Mock()]

    load("turbo", "cuda", "int8_float16")

    assert mock_whisper.call_count == 2
    assert mock_whisper.call_args_list[1][1]["device"] == "cpu"
    assert mock_whisper.call_args_list[1][1]["compute_type"] == "int8"


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_raises_on_non_cuda_error(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """load() raises TranscriptionError for non-CUDA errors."""
    mock_whisper.side_effect = RuntimeError("other error")

    with pytest.raises(TranscriptionError, match="Failed to load model"):
        load("turbo")


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_unload_removes_model_from_cache(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """unload() removes model from cache."""
    mock_whisper.return_value = Mock()

    load("turbo", "cuda", "int8")
    assert len(_model_cache) == 1

    unload("turbo")
    assert len(_model_cache) == 0


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_unload_removes_all_variants(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """unload() removes all cache entries for a model name."""
    mock_whisper.return_value = Mock()

    load("turbo", "cuda", "int8")
    load("turbo", "cpu", "int8")
    load("base", "cuda", "int8")

    unload("turbo")

    assert len(_model_cache) == 1
    assert ("base", "cuda", "int8") in _model_cache


def test_unload_safe_when_model_not_loaded() -> None:
    """unload() is safe to call when model not loaded."""
    unload("nonexistent")


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_unload_all_clears_entire_cache(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """unload_all() clears entire cache."""
    mock_whisper.return_value = Mock()

    load("turbo", "cuda", "int8")
    load("base", "cpu", "int8")
    load("large-v3", "cuda", "float16")

    assert len(_model_cache) == 3

    unload_all()

    assert len(_model_cache) == 0


def test_unload_all_safe_when_cache_empty() -> None:
    """unload_all() is safe when cache is empty."""
    unload_all()


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_is_loaded_returns_true_when_cached(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """is_loaded() returns True when model is cached."""
    mock_whisper.return_value = Mock()

    load("turbo")

    assert is_loaded("turbo") is True


def test_is_loaded_returns_false_when_not_cached() -> None:
    """is_loaded() returns False when model not cached."""
    assert is_loaded("turbo") is False


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_is_loaded_checks_any_variant(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """is_loaded() returns True if any variant is cached."""
    mock_whisper.return_value = Mock()

    load("turbo", "cpu", "int8")

    assert is_loaded("turbo") is True


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_get_returns_cached_model(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """get() returns cached model instance."""
    mock_model = Mock()
    mock_whisper.return_value = mock_model

    load("turbo", "cuda", "int8")

    result = get("turbo", "cuda", "int8")

    assert result is mock_model


def test_get_returns_none_when_not_cached() -> None:
    """get() returns None when model not cached."""
    result = get("turbo", "cuda", "int8")

    assert result is None


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_get_requires_exact_match(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """get() requires exact match of model, device, and compute_type."""
    mock_whisper.return_value = Mock()

    load("turbo", "cuda", "int8")

    assert get("turbo", "cpu", "int8") is None
    assert get("turbo", "cuda", "float16") is None
    assert get("turbo", "cuda", "int8") is not None


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_cpu_fallback_loads_on_cpu(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """_load_cpu_fallback loads model on CPU with int8."""
    mock_model = Mock()
    mock_whisper.return_value = mock_model

    result = _load_cpu_fallback("turbo")

    mock_whisper.assert_called_once_with(
        str(Path("C:/models/turbo")),
        device="cpu",
        compute_type="int8",
    )
    assert result is mock_model


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_cpu_fallback_caches_result(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """_load_cpu_fallback caches the CPU model."""
    mock_whisper.return_value = Mock()

    result1 = _load_cpu_fallback("turbo")
    result2 = _load_cpu_fallback("turbo")

    assert mock_whisper.call_count == 1
    assert result1 is result2


@patch("voicepad_core.inference.model_manager.ensure_model_downloaded", return_value=Path("C:/models/turbo"))
@patch("voicepad_core.inference.model_manager.WhisperModel")
def test_load_cpu_fallback_raises_on_failure(mock_whisper: Mock, mock_ensure: Mock) -> None:
    """_load_cpu_fallback raises TranscriptionError if CPU load fails."""
    mock_whisper.side_effect = Exception("CPU load failed")

    with pytest.raises(TranscriptionError, match="CPU fallback also failed"):
        _load_cpu_fallback("turbo")


@patch("voicepad_core.inference.model_manager.get_config")
def test_warmup_model_skips_when_disabled(mock_get_config: Mock) -> None:
    config = Mock(model_warmup_enabled=False)
    mock_get_config.return_value = config
    model = Mock()

    from voicepad_core.inference.model_manager import _warmup_model

    _warmup_model(model, None)

    model.transcribe.assert_not_called()


@patch("voicepad_core.inference.model_manager.get_config")
def test_warmup_model_uses_configured_settings(mock_get_config: Mock) -> None:
    config = Mock(
        model_warmup_enabled=True,
        model_warmup_duration_s=0.25,
        model_warmup_language="fr",
        model_warmup_beam_size=2,
        model_warmup_vad_filter=True,
    )
    mock_get_config.return_value = config
    model = Mock()
    model.transcribe.return_value = (iter([1]), None)

    from voicepad_core.inference.model_manager import _warmup_model

    _warmup_model(model, None)

    args, kwargs = model.transcribe.call_args
    assert len(args[0]) == 4000
    assert kwargs == {"language": "fr", "beam_size": 2, "vad_filter": True}


@patch("voicepad_core.inference.model_manager.get_config")
def test_warmup_model_failure_is_non_fatal(mock_get_config: Mock) -> None:
    config = Mock(
        model_warmup_enabled=True,
        model_warmup_duration_s=0.25,
        model_warmup_language="fr",
        model_warmup_beam_size=2,
        model_warmup_vad_filter=True,
    )
    mock_get_config.return_value = config
    model = Mock()
    model.transcribe.side_effect = RuntimeError("boom")

    from voicepad_core.inference.model_manager import _warmup_model

    _warmup_model(model, None)
