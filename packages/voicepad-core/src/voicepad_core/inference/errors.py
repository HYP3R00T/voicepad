"""Custom errors for the inference package."""


class TranscriptionError(Exception):
    """Base exception for all transcription failures."""


class AudioTooShortError(TranscriptionError):
    """Raised when audio duration is below the minimum threshold."""


class AudioTooLongWarning(UserWarning):
    """Issued when audio duration exceeds the recommended maximum."""


class ModelNotFoundError(TranscriptionError):
    """Raised when model files are missing and cannot be downloaded."""


class BackendLookupError(TranscriptionError):
    """Raised when a requested inference backend is not registered."""


class BackendUnavailableError(TranscriptionError):
    """Raised when a registered inference backend cannot run."""


class BackendSessionError(TranscriptionError):
    """Raised when an inference backend session cannot be opened or closed safely."""
