from voicepad_core.audio import SUPPORTED_FORMATS
from voicepad_core.audio.constants import DEFAULT_INPUT_CHANNELS, FALLBACK_INPUT_SAMPLE_RATE, PCM_WAV_SUBTYPE
from voicepad_core.audio.errors import (
    AudioConversionDependencyError,
    AudioConversionError,
    AudioFileNotFoundError,
    AudioStreamStateError,
    UnsupportedAudioFormatError,
)


def test_audio_constants_match_expected_defaults() -> None:
    assert DEFAULT_INPUT_CHANNELS == 1
    assert FALLBACK_INPUT_SAMPLE_RATE == 16_000
    assert PCM_WAV_SUBTYPE == "PCM_16"
    assert frozenset({".wav", ".flac", ".ogg", ".mp3", ".m4a", ".mp4"}) == SUPPORTED_FORMATS


def test_audio_errors_preserve_builtin_exception_types() -> None:
    assert issubclass(AudioFileNotFoundError, FileNotFoundError)
    assert issubclass(UnsupportedAudioFormatError, ValueError)
    assert issubclass(AudioConversionDependencyError, RuntimeError)
    assert issubclass(AudioConversionError, RuntimeError)
    assert issubclass(AudioStreamStateError, RuntimeError)
