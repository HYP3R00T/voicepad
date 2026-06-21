"""Custom errors for the inference package."""


class TranscriptionError(Exception):
    """Base exception for all transcription failures."""


class AudioTooShortError(TranscriptionError):
    """Raised when audio duration is below the minimum threshold."""


class AudioTooLongWarning(UserWarning):
    """Issued when audio duration exceeds the recommended maximum."""


class ModelNotFoundError(TranscriptionError):
    """Raised when model files are missing and cannot be downloaded."""
