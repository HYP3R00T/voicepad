"""Tests for voicepad_core.inference.types."""

from __future__ import annotations

import pytest
from voicepad_core.inference.types import Segment, TranscriptionResult, WordTimestamp

# ============================================================================
# WordTimestamp tests
# ============================================================================


def test_word_timestamp_creation() -> None:
    """WordTimestamp can be created with required fields."""
    word = WordTimestamp(
        word="hello",
        start=0.0,
        end=0.5,
        probability=0.95,
    )

    assert word.word == "hello"
    assert word.start == 0.0
    assert word.end == 0.5
    assert word.probability == 0.95


def test_word_timestamp_default_probability() -> None:
    """WordTimestamp has default probability of 0.0."""
    word = WordTimestamp(word="hello", start=0.0, end=0.5)

    assert word.probability == 0.0


def test_word_timestamp_is_frozen() -> None:
    """WordTimestamp is immutable (frozen dataclass)."""
    word = WordTimestamp(word="hello", start=0.0, end=0.5)

    # Verify it's a frozen dataclass by checking __dataclass_fields__
    assert hasattr(word, "__dataclass_fields__")
    # Frozen dataclasses don't allow attribute assignment
    assert word.__class__.__dataclass_params__.frozen  # type: ignore[attr-defined]


# ============================================================================
# Segment tests
# ============================================================================


def test_segment_creation() -> None:
    """Segment can be created with required fields."""
    segment = Segment(
        start=0.0,
        end=1.0,
        text="Hello world",
        avg_logprob=-0.5,
        no_speech_prob=0.1,
    )

    assert segment.start == 0.0
    assert segment.end == 1.0
    assert segment.text == "Hello world"
    assert segment.avg_logprob == -0.5
    assert segment.no_speech_prob == 0.1


def test_segment_default_values() -> None:
    """Segment has default values for optional fields."""
    segment = Segment(start=0.0, end=1.0, text="test")

    assert segment.avg_logprob == 0.0
    assert segment.no_speech_prob == 0.0
    assert segment.words == []


def test_segment_duration() -> None:
    """Segment.duration() returns correct duration."""
    segment = Segment(start=1.5, end=3.7, text="test")

    assert segment.duration() == pytest.approx(2.2)


def test_segment_with_words() -> None:
    """Segment can contain word timestamps."""
    words = [
        WordTimestamp(word="hello", start=0.0, end=0.5),
        WordTimestamp(word="world", start=0.5, end=1.0),
    ]

    segment = Segment(start=0.0, end=1.0, text="hello world", words=words)

    assert len(segment.words) == 2
    assert segment.words[0].word == "hello"
    assert segment.words[1].word == "world"


def test_segment_is_frozen() -> None:
    """Segment is immutable (frozen dataclass)."""
    segment = Segment(start=0.0, end=1.0, text="test")

    # Verify it's a frozen dataclass
    assert hasattr(segment, "__dataclass_fields__")
    assert segment.__class__.__dataclass_params__.frozen  # type: ignore[attr-defined]


# ============================================================================
# TranscriptionResult tests
# ============================================================================


def test_transcription_result_creation() -> None:
    """TranscriptionResult can be created with required fields."""
    segments = [Segment(start=0.0, end=1.0, text="test")]

    result = TranscriptionResult(
        text="test",
        segments=segments,
        language="en",
        language_probability=0.99,
        duration_s=1.0,
        latency_ms=100.0,
        device="cuda",
        compute_type="int8",
    )

    assert result.text == "test"
    assert len(result.segments) == 1
    assert result.language == "en"
    assert result.language_probability == 0.99
    assert result.duration_s == 1.0
    assert result.latency_ms == 100.0
    assert result.device == "cuda"
    assert result.compute_type == "int8"


def test_transcription_result_default_values() -> None:
    """TranscriptionResult has default values for optional fields."""
    result = TranscriptionResult(
        text="test",
        segments=[],
        language="en",
        language_probability=0.99,
        duration_s=1.0,
        latency_ms=100.0,
        device="cuda",
        compute_type="int8",
    )

    assert result.fallback_to_cpu is False
    assert result.avg_confidence == 0.0
    assert result.low_confidence_count == 0


def test_transcription_result_with_fallback() -> None:
    """TranscriptionResult can indicate CPU fallback."""
    result = TranscriptionResult(
        text="test",
        segments=[],
        language="en",
        language_probability=0.99,
        duration_s=1.0,
        latency_ms=100.0,
        device="cpu",
        compute_type="int8",
        fallback_to_cpu=True,
    )

    assert result.fallback_to_cpu is True
    assert result.device == "cpu"


def test_transcription_result_with_quality_metrics() -> None:
    """TranscriptionResult can include quality metrics."""
    result = TranscriptionResult(
        text="test",
        segments=[],
        language="en",
        language_probability=0.99,
        duration_s=1.0,
        latency_ms=100.0,
        device="cuda",
        compute_type="int8",
        avg_confidence=-0.3,
        low_confidence_count=2,
    )

    assert result.avg_confidence == -0.3
    assert result.low_confidence_count == 2


def test_transcription_result_is_frozen() -> None:
    """TranscriptionResult is immutable (frozen dataclass)."""
    result = TranscriptionResult(
        text="test",
        segments=[],
        language="en",
        language_probability=0.99,
        duration_s=1.0,
        latency_ms=100.0,
        device="cuda",
        compute_type="int8",
    )

    # Verify it's a frozen dataclass
    assert hasattr(result, "__dataclass_fields__")
    assert result.__class__.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_transcription_result_empty_segments() -> None:
    """TranscriptionResult can have empty segments list."""
    result = TranscriptionResult(
        text="",
        segments=[],
        language="en",
        language_probability=0.99,
        duration_s=0.5,
        latency_ms=50.0,
        device="cuda",
        compute_type="int8",
    )

    assert result.text == ""
    assert len(result.segments) == 0


def test_transcription_result_multiple_segments() -> None:
    """TranscriptionResult can contain multiple segments."""
    segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="world"),
        Segment(start=2.0, end=3.0, text="test"),
    ]

    result = TranscriptionResult(
        text="Hello world test",
        segments=segments,
        language="en",
        language_probability=0.99,
        duration_s=3.0,
        latency_ms=300.0,
        device="cuda",
        compute_type="int8",
    )

    assert len(result.segments) == 3
    assert result.segments[0].text == "Hello"
    assert result.segments[1].text == "world"
    assert result.segments[2].text == "test"
