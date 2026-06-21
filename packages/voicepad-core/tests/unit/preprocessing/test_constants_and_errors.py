from voicepad_core.preprocessing import TARGET_SAMPLE_RATE
from voicepad_core.preprocessing.constants import MONO_CHANNELS
from voicepad_core.preprocessing.errors import (
    InvalidAudioMetadataError,
    InvalidAudioShapeError,
    PreprocessingError,
)


def test_preprocessing_constants_match_expected_defaults() -> None:
    assert MONO_CHANNELS == 1
    assert TARGET_SAMPLE_RATE == 16_000


def test_preprocessing_errors_preserve_value_error_type() -> None:
    assert issubclass(PreprocessingError, Exception)
    assert issubclass(InvalidAudioMetadataError, ValueError)
    assert issubclass(InvalidAudioShapeError, ValueError)
