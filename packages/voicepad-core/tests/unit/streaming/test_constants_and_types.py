from voicepad_core.streaming.constants import MAX_CHUNK_S, MIN_CHUNK_S, OVERLAP_S, POLL_INTERVAL_S, SILENCE_THRESHOLD_MS
from voicepad_core.streaming.errors import StreamingConfigurationError, StreamingError, StreamingRecorderError
from voicepad_core.streaming.types import ChunkResult


def test_streaming_constants_match_expected_defaults() -> None:
    assert MIN_CHUNK_S == 15.0
    assert MAX_CHUNK_S == 29.0
    assert OVERLAP_S == 0.5
    assert POLL_INTERVAL_S == 0.3
    assert SILENCE_THRESHOLD_MS == 1000


def test_streaming_errors_preserve_hierarchy() -> None:
    assert issubclass(StreamingConfigurationError, StreamingError)
    assert issubclass(StreamingRecorderError, StreamingError)


def test_chunk_result_defaults_are_stable() -> None:
    result = ChunkResult(index=1, text="hello")

    assert result.index == 1
    assert result.text == "hello"
    assert result.is_final is False
    assert result.segments == []
