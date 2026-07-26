"""Tests for voicepad_core.inference.errors."""

from __future__ import annotations

import pytest
from voicepad_core.inference.errors import (
    AudioTooLongWarning,
    AudioTooShortError,
    BackendLookupError,
    BackendSessionError,
    BackendUnavailableError,
    ModelNotFoundError,
    TranscriptionError,
)


def test_transcription_error_is_exception() -> None:
    error = TranscriptionError("Test error")

    assert isinstance(error, Exception)
    assert str(error) == "Test error"


def test_audio_too_short_error_inheritance() -> None:
    error = AudioTooShortError("Audio too short")

    assert isinstance(error, AudioTooShortError)
    assert isinstance(error, TranscriptionError)
    assert isinstance(error, Exception)
    assert str(error) == "Audio too short"


def test_audio_too_long_warning_inheritance() -> None:
    warning = AudioTooLongWarning("Audio too long")

    assert isinstance(warning, AudioTooLongWarning)
    assert isinstance(warning, UserWarning)
    assert str(warning) == "Audio too long"


def test_model_not_found_error_inheritance() -> None:
    error = ModelNotFoundError("Model not found")

    assert isinstance(error, ModelNotFoundError)
    assert isinstance(error, TranscriptionError)
    assert isinstance(error, Exception)
    assert str(error) == "Model not found"


def test_errors_can_be_raised_and_caught() -> None:
    with pytest.raises(TranscriptionError, match="General error"):
        raise TranscriptionError("General error")

    with pytest.raises(AudioTooShortError, match="Too short"):
        raise AudioTooShortError("Too short")

    with pytest.raises(ModelNotFoundError, match="Not found"):
        raise ModelNotFoundError("Not found")


def test_warning_can_be_issued() -> None:
    with pytest.warns(AudioTooLongWarning, match="Long audio"):
        import warnings

        warnings.warn("Long audio", AudioTooLongWarning, stacklevel=1)


def test_error_hierarchy() -> None:
    errors = [
        TranscriptionError("Base error"),
        AudioTooShortError("Short audio"),
        ModelNotFoundError("Missing model"),
        BackendLookupError("Missing backend"),
        BackendUnavailableError("Unavailable backend"),
        BackendSessionError("Invalid session"),
    ]

    for error in errors:
        with pytest.raises(TranscriptionError):
            raise error
