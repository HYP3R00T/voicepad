"""Tests for LocalAgreement two-pass verification."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from voicepad_core.postprocessing.agreement import _compare_tokens, apply_local_agreement


class TestCompareTokens:
    """Tests for _compare_tokens function."""

    def test_identical_texts_all_agreed(self) -> None:
        """Identical texts should have all tokens agreed."""
        text1 = "hello world this is a test"
        text2 = "hello world this is a test"
        result = _compare_tokens(text1, text2)
        assert result == text1

    def test_case_insensitive_comparison(self) -> None:
        """Token comparison should be case-insensitive."""
        text1 = "Hello World"
        text2 = "hello world"
        result = _compare_tokens(text1, text2)
        assert result == "Hello World"  # Preserves first text's case

    def test_different_tokens_not_agreed(self) -> None:
        """Different tokens should not be included."""
        text1 = "hello world test"
        text2 = "hello earth test"
        result = _compare_tokens(text1, text2)
        assert result == "hello test"  # "world" vs "earth" disagreed

    def test_different_lengths_shorter_wins(self) -> None:
        """When texts have different lengths, only compare up to shorter."""
        text1 = "hello world test extra"
        text2 = "hello world test"
        result = _compare_tokens(text1, text2)
        assert result == "hello world test"

    def test_empty_first_text(self) -> None:
        """Empty first text should return empty."""
        result = _compare_tokens("", "hello world")
        assert result == ""

    def test_empty_second_text(self) -> None:
        """Empty second text should return empty."""
        result = _compare_tokens("hello world", "")
        assert result == ""

    def test_both_empty(self) -> None:
        """Both empty should return empty."""
        result = _compare_tokens("", "")
        assert result == ""

    def test_single_word_match(self) -> None:
        """Single word that matches."""
        result = _compare_tokens("hello", "hello")
        assert result == "hello"

    def test_single_word_mismatch(self) -> None:
        """Single word that doesn't match."""
        result = _compare_tokens("hello", "world")
        assert result == ""

    def test_partial_agreement(self) -> None:
        """Partial agreement in longer text."""
        text1 = "the quick brown fox jumps"
        text2 = "the fast brown cat jumps"
        result = _compare_tokens(text1, text2)
        assert result == "the brown jumps"

    def test_preserves_first_text_case(self) -> None:
        """Should preserve the case from first text."""
        text1 = "The Quick BROWN fox"
        text2 = "the quick brown fox"
        result = _compare_tokens(text1, text2)
        assert result == "The Quick BROWN fox"

    def test_mixed_case_disagreement(self) -> None:
        """Mixed case with some disagreements."""
        text1 = "Hello WORLD test"
        text2 = "hello earth TEST"
        result = _compare_tokens(text1, text2)
        assert result == "Hello test"


class TestApplyLocalAgreement:
    """Tests for apply_local_agreement function."""

    @patch("voicepad_core.inference.transcribe")
    def test_runs_second_pass_and_compares(self, mock_transcribe: Mock) -> None:
        """Should run second transcription pass and compare results."""
        audio = np.array([0.1, 0.2, 0.3])

        # First result (passed in)
        first_result = Mock()
        first_result.text = "hello world test"
        first_result.segments = []
        first_result.language = "en"
        first_result.language_probability = 0.95
        first_result.duration_s = 3.0
        first_result.latency_ms = 100.0
        first_result.device = "cuda"
        first_result.compute_type = "float16"
        first_result.fallback_to_cpu = False
        first_result.avg_confidence = 0.9
        first_result.low_confidence_count = 0

        # Second result (from mock)
        second_result = Mock()
        second_result.text = "hello world test"  # Same as first
        second_result.latency_ms = 110.0

        mock_transcribe.return_value = second_result

        result = apply_local_agreement(
            audio=audio,
            first_result=first_result,
            model_name="turbo",
            device="cuda",
            compute_type="float16",
            language="en",
        )

        # Verify second pass was called
        mock_transcribe.assert_called_once_with(
            audio,
            model_name="turbo",
            device="cuda",
            compute_type="float16",
            language="en",
        )

        # Verify result
        assert result.text == "hello world test"
        assert result.latency_ms == 210.0  # Sum of both passes

    @patch("voicepad_core.inference.transcribe")
    def test_filters_disagreed_tokens(self, mock_transcribe: Mock) -> None:
        """Should filter out tokens that don't agree."""
        audio = np.array([0.1, 0.2, 0.3])

        first_result = Mock()
        first_result.text = "hello world test"
        first_result.segments = []
        first_result.language = "en"
        first_result.language_probability = 0.95
        first_result.duration_s = 3.0
        first_result.latency_ms = 100.0
        first_result.device = "cuda"
        first_result.compute_type = "float16"
        first_result.fallback_to_cpu = False
        first_result.avg_confidence = 0.9
        first_result.low_confidence_count = 0

        second_result = Mock()
        second_result.text = "hello earth test"  # "world" -> "earth"
        second_result.latency_ms = 110.0

        mock_transcribe.return_value = second_result

        result = apply_local_agreement(
            audio=audio,
            first_result=first_result,
            model_name="turbo",
            device="cuda",
            compute_type="float16",
            language="en",
        )

        # "world" should be filtered out
        assert result.text == "hello test"

    @patch("voicepad_core.inference.transcribe")
    def test_preserves_first_result_metadata(self, mock_transcribe: Mock) -> None:
        """Should preserve metadata from first result."""
        audio = np.array([0.1, 0.2, 0.3])

        first_result = Mock()
        first_result.text = "hello"
        first_result.segments = [Mock(), Mock()]
        first_result.language = "en"
        first_result.language_probability = 0.95
        first_result.duration_s = 3.0
        first_result.latency_ms = 100.0
        first_result.device = "cuda"
        first_result.compute_type = "float16"
        first_result.fallback_to_cpu = False
        first_result.avg_confidence = 0.9
        first_result.low_confidence_count = 2

        second_result = Mock()
        second_result.text = "hello"
        second_result.latency_ms = 110.0

        mock_transcribe.return_value = second_result

        result = apply_local_agreement(
            audio=audio,
            first_result=first_result,
            model_name="turbo",
            device="cuda",
            compute_type="float16",
            language="en",
        )

        # Verify metadata preserved
        assert result.segments == first_result.segments
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert result.duration_s == 3.0
        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback_to_cpu is False
        assert result.avg_confidence == 0.9
        assert result.low_confidence_count == 2

    @patch("voicepad_core.inference.transcribe")
    def test_sums_latencies(self, mock_transcribe: Mock) -> None:
        """Should sum latencies from both passes."""
        audio = np.array([0.1, 0.2, 0.3])

        first_result = Mock()
        first_result.text = "hello"
        first_result.segments = []
        first_result.language = "en"
        first_result.language_probability = 0.95
        first_result.duration_s = 3.0
        first_result.latency_ms = 123.45
        first_result.device = "cuda"
        first_result.compute_type = "float16"
        first_result.fallback_to_cpu = False
        first_result.avg_confidence = 0.9
        first_result.low_confidence_count = 0

        second_result = Mock()
        second_result.text = "hello"
        second_result.latency_ms = 234.56

        mock_transcribe.return_value = second_result

        result = apply_local_agreement(
            audio=audio,
            first_result=first_result,
            model_name="turbo",
            device="cuda",
            compute_type="float16",
            language="en",
        )

        assert result.latency_ms == pytest.approx(358.01)

    @patch("voicepad_core.inference.transcribe")
    def test_empty_agreement(self, mock_transcribe: Mock) -> None:
        """Should handle case where no tokens agree."""
        audio = np.array([0.1, 0.2, 0.3])

        first_result = Mock()
        first_result.text = "hello world"
        first_result.segments = []
        first_result.language = "en"
        first_result.language_probability = 0.95
        first_result.duration_s = 3.0
        first_result.latency_ms = 100.0
        first_result.device = "cuda"
        first_result.compute_type = "float16"
        first_result.fallback_to_cpu = False
        first_result.avg_confidence = 0.9
        first_result.low_confidence_count = 0

        second_result = Mock()
        second_result.text = "completely different"
        second_result.latency_ms = 110.0

        mock_transcribe.return_value = second_result

        result = apply_local_agreement(
            audio=audio,
            first_result=first_result,
            model_name="turbo",
            device="cuda",
            compute_type="float16",
            language="en",
        )

        # No agreement means empty text
        assert result.text == ""
