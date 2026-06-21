class StreamingError(Exception):
    """Base class for streaming package errors."""


class StreamingConfigurationError(StreamingError, ValueError):
    """Raised when streaming configuration values are invalid."""


class StreamingRecorderError(StreamingError, RuntimeError):
    """Raised when the recorder cannot provide streaming audio."""


__all__ = ["StreamingError", "StreamingConfigurationError", "StreamingRecorderError"]
