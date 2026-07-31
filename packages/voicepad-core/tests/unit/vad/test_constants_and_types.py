from voicepad_core.vad import InvalidVADSampleRateError, SpeechSegment, VADModelDownloadError


def test_speech_segment_stores_boundaries() -> None:
    segment = SpeechSegment(start=0.25, end=0.75)

    assert (segment.start, segment.end) == (0.25, 0.75)


def test_vad_errors_preserve_hierarchy() -> None:
    assert issubclass(InvalidVADSampleRateError, ValueError)
    assert issubclass(VADModelDownloadError, RuntimeError)
