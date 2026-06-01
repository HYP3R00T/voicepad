# inference/exceptions.py

"""Custom exceptions for the inference package."""


class TranscriptionError(Exception):
    """Base exception for all transcription failures."""


class AudioTooShortError(TranscriptionError):
    """Raised when audio duration is below the minimum threshold.

    The caller should discard the audio or ask the user to speak longer.
    """


class AudioTooLongWarning(UserWarning):
    """Issued when audio duration exceeds the recommended maximum.

    Transcription still proceeds but may be slow or inaccurate.
    """


class ModelNotFoundError(TranscriptionError):
    """Raised when the model files are missing and cannot be downloaded.

    This wraps network errors, HuggingFace errors, or missing local paths.
    """
