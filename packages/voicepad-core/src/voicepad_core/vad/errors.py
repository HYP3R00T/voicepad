class VADError(Exception):
    """Base class for VAD package errors."""


class InvalidVADSampleRateError(VADError, ValueError):
    """Raised when VAD receives audio at an unsupported sample rate."""


class VADModelDownloadError(VADError, RuntimeError):
    """Raised when the Silero VAD model cannot be downloaded."""


__all__ = ["VADError", "InvalidVADSampleRateError", "VADModelDownloadError"]
