import pytest
from voicepad_core.vad.errors import InvalidVADSampleRateError, VADError, VADModelDownloadError
from voicepad_core.vad.types import SpeechSegment


def test_speech_segment_helpers_work() -> None:
    segment = SpeechSegment(start=0.25, end=0.75)

    assert segment.duration() == pytest.approx(0.5)
    assert segment.to_sample_indices(16_000) == (4000, 12000)


def test_vad_errors_preserve_hierarchy() -> None:
    assert issubclass(InvalidVADSampleRateError, VADError)
    assert issubclass(VADModelDownloadError, VADError)
