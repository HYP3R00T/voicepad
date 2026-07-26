from voicepad_core.streaming import ChunkResult, StreamingConfigurationError


def test_streaming_errors_preserve_hierarchy() -> None:
    assert issubclass(StreamingConfigurationError, ValueError)


def test_chunk_result_defaults_are_stable() -> None:
    result = ChunkResult(index=1, text="hello")

    assert result.index == 1
    assert result.text == "hello"
    assert result.is_final is False
    assert result.segments == []
