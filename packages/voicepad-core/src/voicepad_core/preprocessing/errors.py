class PreprocessingError(Exception):
    """Base class for preprocessing errors."""


class InvalidAudioMetadataError(ValueError, PreprocessingError):
    """Raised when sample-rate or channel metadata is invalid."""


class InvalidAudioShapeError(ValueError, PreprocessingError):
    """Raised when audio data shape does not match the declared metadata."""
