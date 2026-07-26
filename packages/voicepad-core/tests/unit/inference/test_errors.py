"""Tests for voicepad_core.inference.errors."""

from __future__ import annotations

import pytest
from voicepad_core.inference.errors import (
    AudioTooShortError,
    BackendLookupError,
    BackendSessionError,
    BackendUnavailableError,
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


def test_errors_can_be_raised_and_caught() -> None:
    with pytest.raises(TranscriptionError, match="General error"):
        raise TranscriptionError("General error")

    with pytest.raises(AudioTooShortError, match="Too short"):
        raise AudioTooShortError("Too short")


def test_error_hierarchy() -> None:
    errors = [
        TranscriptionError("Base error"),
        AudioTooShortError("Short audio"),
        BackendLookupError("Missing backend"),
        BackendUnavailableError("Unavailable backend"),
        BackendSessionError("Invalid session"),
    ]

    for error in errors:
        with pytest.raises(TranscriptionError):
            raise error
