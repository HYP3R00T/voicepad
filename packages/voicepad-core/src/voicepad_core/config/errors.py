class ConfigError(ValueError):
    """Base class for config package validation errors."""


class UnknownTranscriptionModelError(ConfigError):
    """Raised when transcription_model is not one of the supported values."""


__all__ = ["ConfigError", "UnknownTranscriptionModelError"]
