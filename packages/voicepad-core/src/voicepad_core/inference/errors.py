class TranscriptionError(Exception):
    """Base exception for all transcription failures."""


class AudioTooShortError(TranscriptionError):
    """Raised when audio duration is below the minimum threshold."""


class BackendLookupError(TranscriptionError):
    """Raised when a requested inference backend is not registered."""


class BackendUnavailableError(TranscriptionError):
    """Raised when a registered inference backend cannot run."""


class BackendSessionError(TranscriptionError):
    """Raised when an inference backend session cannot be opened or closed safely."""
