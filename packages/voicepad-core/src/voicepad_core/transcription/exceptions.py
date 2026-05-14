"""Exceptions for transcription operations."""


class TranscriptionError(Exception):
    """Base exception for transcription failures."""


class AudioTooShortError(TranscriptionError):
    """Audio duration below minimum threshold."""


class AudioTooLongWarning(UserWarning):
    """Audio duration exceeds recommended maximum."""
