class AudioError(Exception):
    """Base class for audio package errors."""


class AudioFileNotFoundError(FileNotFoundError, AudioError):
    """Raised when an audio file path does not exist."""


class UnsupportedAudioFormatError(ValueError, AudioError):
    """Raised when an audio file extension is not supported."""


class AudioConversionDependencyError(RuntimeError, AudioError):
    """Raised when a required audio conversion dependency is unavailable."""


class AudioConversionError(RuntimeError, AudioError):
    """Raised when audio conversion fails."""


class AudioStreamStateError(RuntimeError, AudioError):
    """Raised when microphone stream lifecycle methods are misused."""


class AudioWriteBackpressureError(RuntimeError, AudioError):
    """Raised when durable audio writing cannot keep pace with capture."""
