"""Tests for segment filtering."""

from __future__ import annotations

from unittest.mock import Mock

from voicepad_core.postprocessing.filters import filter_segments


class TestFilterSegments:
    """Tests for filter_segments function."""

    def test_empty_segments_returns_empty(self) -> None:
        """Empty segment list should return empty."""
        result = filter_segments(iter([]), duration_s=10.0)
        assert result == []

    def test_valid_segment_kept(self) -> None:
        """Valid segment should be kept."""
        mock_seg = Mock()
        mock_seg.start = 1.0
        mock_seg.end = 3.0
        mock_seg.text = "  hello world  "
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1
        mock_seg.words = []

        result = filter_segments(iter([mock_seg]), duration_s=10.0)

        assert len(result) == 1
        assert result[0].start == 1.0
        assert result[0].end == 3.0
        assert result[0].text == "hello world"  # Stripped
        assert result[0].avg_logprob == -0.5
        assert result[0].no_speech_prob == 0.1

    def test_segment_at_boundary_dropped(self) -> None:
        """Segment starting at or beyond duration should be dropped."""
        mock_seg = Mock()
        mock_seg.start = 10.0
        mock_seg.end = 12.0
        mock_seg.text = "hello"
        mock_seg.no_speech_prob = 0.1

        result = filter_segments(iter([mock_seg]), duration_s=10.0)
        assert result == []

    def test_segment_beyond_boundary_dropped(self) -> None:
        """Segment starting beyond duration should be dropped."""
        mock_seg = Mock()
        mock_seg.start = 11.0
        mock_seg.end = 13.0
        mock_seg.text = "hello"
        mock_seg.no_speech_prob = 0.1

        result = filter_segments(iter([mock_seg]), duration_s=10.0)
        assert result == []

    def test_short_segment_near_tail_dropped(self) -> None:
        """Short segment (<0.5s) near tail (last 1s) should be dropped."""
        mock_seg = Mock()
        mock_seg.start = 9.6
        mock_seg.end = 9.9  # 0.3s duration, within last 1s
        mock_seg.text = "hello"
        mock_seg.no_speech_prob = 0.1

        result = filter_segments(iter([mock_seg]), duration_s=10.0)
        assert result == []

    def test_short_segment_not_near_tail_kept(self) -> None:
        """Short segment not near tail should be kept."""
        mock_seg = Mock()
        mock_seg.start = 5.0
        mock_seg.end = 5.3  # 0.3s duration, not near tail
        mock_seg.text = "hello"
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1
        mock_seg.words = []

        result = filter_segments(iter([mock_seg]), duration_s=10.0)
        assert len(result) == 1

    def test_long_segment_near_tail_kept(self) -> None:
        """Long segment (>=0.5s) near tail should be kept."""
        mock_seg = Mock()
        mock_seg.start = 9.2
        mock_seg.end = 9.8  # 0.6s duration, within last 1s
        mock_seg.text = "hello world"
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1
        mock_seg.words = []

        result = filter_segments(iter([mock_seg]), duration_s=10.0)
        assert len(result) == 1

    def test_high_no_speech_prob_dropped(self) -> None:
        """Segment with no_speech_prob > 0.9 should be dropped."""
        mock_seg = Mock()
        mock_seg.start = 1.0
        mock_seg.end = 3.0
        mock_seg.text = "hello"
        mock_seg.no_speech_prob = 0.95

        result = filter_segments(iter([mock_seg]), duration_s=10.0)
        assert result == []

    def test_no_speech_prob_at_threshold_dropped(self) -> None:
        """Segment with no_speech_prob exactly at 0.9 should be kept."""
        mock_seg = Mock()
        mock_seg.start = 1.0
        mock_seg.end = 3.0
        mock_seg.text = "hello"
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.9
        mock_seg.words = []

        result = filter_segments(iter([mock_seg]), duration_s=10.0)
        assert len(result) == 1

    def test_segment_end_clamped_to_duration(self) -> None:
        """Segment end should be clamped to duration."""
        mock_seg = Mock()
        mock_seg.start = 8.0
        mock_seg.end = 12.0  # Beyond duration
        mock_seg.text = "hello"
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1
        mock_seg.words = []

        result = filter_segments(iter([mock_seg]), duration_s=10.0)

        assert len(result) == 1
        assert result[0].end == 10.0  # Clamped

    def test_multiple_segments_mixed(self) -> None:
        """Multiple segments with some kept and some dropped."""
        seg1 = Mock()
        seg1.start = 1.0
        seg1.end = 2.0
        seg1.text = "valid"
        seg1.avg_logprob = -0.5
        seg1.no_speech_prob = 0.1
        seg1.words = []

        seg2 = Mock()
        seg2.start = 11.0  # Beyond boundary
        seg2.end = 12.0
        seg2.text = "invalid"
        seg2.no_speech_prob = 0.1

        seg3 = Mock()
        seg3.start = 3.0
        seg3.end = 4.0
        seg3.text = "also valid"
        seg3.avg_logprob = -0.6
        seg3.no_speech_prob = 0.2
        seg3.words = []

        seg4 = Mock()
        seg4.start = 5.0
        seg4.end = 6.0
        seg4.text = "no speech"
        seg4.no_speech_prob = 0.95  # Too high

        result = filter_segments(iter([seg1, seg2, seg3, seg4]), duration_s=10.0)

        assert len(result) == 2
        assert result[0].text == "valid"
        assert result[1].text == "also valid"

    def test_segment_with_words_attribute(self) -> None:
        """Segment with words attribute should preserve it."""
        mock_seg = Mock()
        mock_seg.start = 1.0
        mock_seg.end = 2.0
        mock_seg.text = "hello"
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1
        mock_seg.words = [Mock(), Mock()]

        result = filter_segments(iter([mock_seg]), duration_s=10.0)

        assert len(result) == 1
        assert len(result[0].words) == 2

    def test_segment_without_words_attribute(self) -> None:
        """Segment without words attribute should get empty list."""
        mock_seg = Mock(spec=["start", "end", "text", "avg_logprob", "no_speech_prob"])
        mock_seg.start = 1.0
        mock_seg.end = 2.0
        mock_seg.text = "hello"
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1

        result = filter_segments(iter([mock_seg]), duration_s=10.0)

        assert len(result) == 1
        assert result[0].words == []

    def test_text_stripping(self) -> None:
        """Text should be stripped of leading/trailing whitespace."""
        mock_seg = Mock()
        mock_seg.start = 1.0
        mock_seg.end = 2.0
        mock_seg.text = "  \n  hello world  \t  "
        mock_seg.avg_logprob = -0.5
        mock_seg.no_speech_prob = 0.1
        mock_seg.words = []

        result = filter_segments(iter([mock_seg]), duration_s=10.0)

        assert result[0].text == "hello world"
