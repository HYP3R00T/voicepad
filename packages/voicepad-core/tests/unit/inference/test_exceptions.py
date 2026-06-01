"""Tests for voicepad_core.inference.exceptions."""

from __future__ import annotations

import pytest
from voicepad_core.inference.exceptions import (
    AudioTooLongWarning,
    AudioTooShortError,
    ModelNotFoundError,
    TranscriptionError,
)

# ============================================================================
# TranscriptionError tests
# ============================================================================


def test_transcription_error_is_exception() -> None:
    """TranscriptionError is an Exception."""
    error = TranscriptionError("test error")
    assert isinstance(error, Exception)


def test_transcription_error_message() -> None:
    """TranscriptionError preserves error message."""
    error = TranscriptionError("custom message")
    assert str(error) == "custom message"


def test_transcription_error_can_be_raised() -> None:
    """TranscriptionError can be raised and caught."""
    with pytest.raises(TranscriptionError, match="test"):
        raise TranscriptionError("test")


# ============================================================================
# AudioTooShortError tests
# ============================================================================


def test_audio_too_short_error_is_transcription_error() -> None:
    """AudioTooShortError is a TranscriptionError."""
    error = AudioTooShortError("too short")
    assert isinstance(error, TranscriptionError)
    assert isinstance(error, Exception)


def test_audio_too_short_error_message() -> None:
    """AudioTooShortError preserves error message."""
    error = AudioTooShortError("Audio is 0.3s — below minimum 0.5s")
    assert "0.3s" in str(error)
    assert "0.5s" in str(error)


def test_audio_too_short_error_can_be_raised() -> None:
    """AudioTooShortError can be raised and caught."""
    with pytest.raises(AudioTooShortError, match="too short"):
        raise AudioTooShortError("too short")


def test_audio_too_short_error_caught_as_transcription_error() -> None:
    """AudioTooShortError can be caught as TranscriptionError."""
    with pytest.raises(TranscriptionError):
        raise AudioTooShortError("too short")


# ============================================================================
# AudioTooLongWarning tests
# ============================================================================


def test_audio_too_long_warning_is_user_warning() -> None:
    """AudioTooLongWarning is a UserWarning."""
    warning = AudioTooLongWarning("too long")
    assert isinstance(warning, UserWarning)


def test_audio_too_long_warning_message() -> None:
    """AudioTooLongWarning preserves warning message."""
    warning = AudioTooLongWarning("Audio exceeds 30s")
    assert "30s" in str(warning)


def test_audio_too_long_warning_can_be_issued() -> None:
    """AudioTooLongWarning can be issued with warnings.warn."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn("Audio too long", AudioTooLongWarning, stacklevel=2)

        assert len(w) == 1
        assert issubclass(w[0].category, AudioTooLongWarning)


# ============================================================================
# ModelNotFoundError tests
# ============================================================================


def test_model_not_found_error_is_transcription_error() -> None:
    """ModelNotFoundError is a TranscriptionError."""
    error = ModelNotFoundError("model not found")
    assert isinstance(error, TranscriptionError)
    assert isinstance(error, Exception)


def test_model_not_found_error_message() -> None:
    """ModelNotFoundError preserves error message."""
    error = ModelNotFoundError("Failed to download model 'turbo'")
    assert "turbo" in str(error)
    assert "download" in str(error)


def test_model_not_found_error_can_be_raised() -> None:
    """ModelNotFoundError can be raised and caught."""
    with pytest.raises(ModelNotFoundError, match="not found"):
        raise ModelNotFoundError("not found")


def test_model_not_found_error_caught_as_transcription_error() -> None:
    """ModelNotFoundError can be caught as TranscriptionError."""
    with pytest.raises(TranscriptionError):
        raise ModelNotFoundError("not found")


# ============================================================================
# Exception hierarchy tests
# ============================================================================


def test_exception_hierarchy() -> None:
    """Test the exception hierarchy is correct."""
    # TranscriptionError is base
    assert issubclass(TranscriptionError, Exception)

    # AudioTooShortError inherits from TranscriptionError
    assert issubclass(AudioTooShortError, TranscriptionError)
    assert issubclass(AudioTooShortError, Exception)

    # ModelNotFoundError inherits from TranscriptionError
    assert issubclass(ModelNotFoundError, TranscriptionError)
    assert issubclass(ModelNotFoundError, Exception)

    # AudioTooLongWarning is separate (UserWarning)
    assert issubclass(AudioTooLongWarning, UserWarning)
    assert not issubclass(AudioTooLongWarning, TranscriptionError)


def test_catching_base_exception_catches_all() -> None:
    """Catching TranscriptionError catches all transcription-related errors."""
    errors = [
        TranscriptionError("base"),
        AudioTooShortError("too short"),
        ModelNotFoundError("not found"),
    ]

    for error in errors:
        with pytest.raises(TranscriptionError):
            raise error


def test_specific_exception_catching() -> None:
    """Each exception can be caught specifically."""
    with pytest.raises(AudioTooShortError):
        raise AudioTooShortError("too short")

    with pytest.raises(ModelNotFoundError):
        raise ModelNotFoundError("not found")

    with pytest.raises(TranscriptionError):
        raise TranscriptionError("generic")
