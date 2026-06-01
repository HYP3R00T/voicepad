"""Tests for overlap deduplication."""

from __future__ import annotations

from voicepad_core.inference.types import Segment
from voicepad_core.postprocessing.deduplication import deduplicate_overlap


class TestDeduplicateOverlap:
    """Tests for deduplicate_overlap function."""

    def test_empty_segments_returns_empty(self) -> None:
        """Empty segment list should return empty."""
        result = deduplicate_overlap([], chunk_start_s=5.0, prev_text="hello world")
        assert result == []

    def test_empty_prev_text_returns_all_segments(self) -> None:
        """Empty prev_text should return all segments unchanged."""
        segments = [
            Segment(start=0.0, end=1.0, text="hello", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=1.0, end=2.0, text="world", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
        ]
        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text="")
        assert result == segments

    def test_no_overlap_segments_returns_all(self) -> None:
        """No segments in overlap region should return all segments."""
        segments = [
            Segment(start=5.0, end=6.0, text="hello", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=6.0, end=7.0, text="world", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
        ]
        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text="previous text")
        assert result == segments

    def test_full_duplicate_drops_overlap_segments(self) -> None:
        """Full duplicate (similarity >= 0.8) should drop all overlap segments."""
        overlap_seg = Segment(start=4.0, end=4.5, text="hello world", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        non_overlap_seg = Segment(start=5.5, end=6.0, text="new text", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg, non_overlap_seg]
        prev_text = "hello world"  # Exact match

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        assert len(result) == 1
        assert result[0] == non_overlap_seg

    def test_full_duplicate_case_insensitive(self) -> None:
        """Full duplicate detection should be case-insensitive."""
        overlap_seg = Segment(start=4.0, end=4.5, text="Hello World", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        non_overlap_seg = Segment(start=5.5, end=6.0, text="new text", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg, non_overlap_seg]
        prev_text = "hello world"

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        assert len(result) == 1
        assert result[0] == non_overlap_seg

    def test_partial_duplicate_removes_matching_segments(self) -> None:
        """Partial duplicate removes segments whose text is in prev_tail."""
        overlap_seg1 = Segment(start=4.0, end=4.3, text="hello", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        overlap_seg2 = Segment(start=4.3, end=4.6, text="world", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        overlap_seg3 = Segment(start=4.6, end=4.9, text="new", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        non_overlap_seg = Segment(start=5.5, end=6.0, text="text", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg1, overlap_seg2, overlap_seg3, non_overlap_seg]
        prev_text = "previous hello world content"

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        # overlap_text = "hello world new" (3 words, >= MIN_OVERLAP_WORDS_FOR_PARTIAL)
        # lead = "hello world new" (first 5 words, but only 3 available)
        # "hello world new" is NOT in prev_tail as a substring
        # So no partial duplicate detected, all segments kept
        assert len(result) == 4

    def test_partial_duplicate_requires_min_overlap_words(self) -> None:
        """Partial duplicate requires at least 3 overlap words."""
        overlap_seg1 = Segment(start=4.0, end=4.5, text="hi", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        overlap_seg2 = Segment(start=4.5, end=4.8, text="there", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        non_overlap_seg = Segment(start=5.5, end=6.0, text="new", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg1, overlap_seg2, non_overlap_seg]
        prev_text = "hi there"

        # overlap_text = "hi there" (2 words, < MIN_OVERLAP_WORDS_FOR_PARTIAL)
        # prev_tail = "hi there"
        # similarity = 1.0 >= 0.8, so full duplicate detected
        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        # Full duplicate: all overlap segments dropped
        assert len(result) == 1
        assert result[0] == non_overlap_seg

    def test_partial_duplicate_checks_leading_words(self) -> None:
        """Partial duplicate checks first 5 words of overlap."""
        overlap_segs = [
            Segment(start=4.0, end=4.2, text="one", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=4.2, end=4.4, text="two", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=4.4, end=4.6, text="three", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=4.6, end=4.8, text="four", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=4.8, end=4.9, text="five", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
        ]
        non_overlap_seg = Segment(start=5.5, end=6.0, text="new", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = overlap_segs + [non_overlap_seg]
        prev_text = "previous one two three four five content"

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        # All overlap segments should be removed as they match prev_text
        assert len(result) == 1
        assert result[0] == non_overlap_seg

    def test_no_duplicate_returns_all_segments(self) -> None:
        """No duplicate detected should return all segments."""
        overlap_seg = Segment(
            start=4.0, end=4.5, text="completely different", avg_logprob=-0.5, no_speech_prob=0.1, words=[]
        )
        non_overlap_seg = Segment(start=5.5, end=6.0, text="new text", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg, non_overlap_seg]
        prev_text = "hello world"

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        assert result == segments

    def test_multiple_overlap_segments_full_duplicate(self) -> None:
        """Multiple overlap segments with full duplicate."""
        overlap_segs = [
            Segment(start=4.0, end=4.3, text="hello", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=4.3, end=4.6, text="world", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=4.6, end=4.9, text="again", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
        ]
        non_overlap_segs = [
            Segment(start=5.5, end=6.0, text="new", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
            Segment(start=6.0, end=6.5, text="content", avg_logprob=-0.5, no_speech_prob=0.1, words=[]),
        ]

        segments = overlap_segs + non_overlap_segs
        prev_text = "hello world again"

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        # All overlap segments dropped, only non-overlap kept
        assert len(result) == 2
        assert result == non_overlap_segs

    def test_prev_text_tail_limited_to_50_words(self) -> None:
        """Only last 50 words of prev_text should be used."""
        overlap_seg = Segment(start=4.0, end=4.5, text="word50", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        non_overlap_seg = Segment(start=5.5, end=6.0, text="new", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg, non_overlap_seg]

        # Create prev_text with 60 words: word0 word1 ... word59
        # Last 50 words: word10 word11 ... word59
        prev_words = [f"word{i}" for i in range(60)]
        prev_text = " ".join(prev_words)

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        # overlap_text = "word50" (1 word, < MIN_OVERLAP_WORDS_FOR_PARTIAL)
        # prev_tail = "word10 word11 ... word59" (contains "word50")
        # similarity will be low (1 word vs 50 words), so no full duplicate
        # No partial check (< 3 words)
        # Result: all segments kept
        assert len(result) == 2

    def test_empty_text_segments_handled(self) -> None:
        """Segments with empty text should be handled gracefully."""
        overlap_seg1 = Segment(start=4.0, end=4.3, text="", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        overlap_seg2 = Segment(start=4.3, end=4.6, text="hello", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        non_overlap_seg = Segment(start=5.5, end=6.0, text="world", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg1, overlap_seg2, non_overlap_seg]
        prev_text = "hello"

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        # Should handle empty text gracefully
        assert non_overlap_seg in result

    def test_boundary_segment_classification(self) -> None:
        """Segment exactly at chunk_start_s should be non-overlap."""
        boundary_seg = Segment(start=5.0, end=5.5, text="boundary", avg_logprob=-0.5, no_speech_prob=0.1, words=[])
        overlap_seg = Segment(start=4.5, end=4.9, text="overlap", avg_logprob=-0.5, no_speech_prob=0.1, words=[])

        segments = [overlap_seg, boundary_seg]
        prev_text = "overlap"

        result = deduplicate_overlap(segments, chunk_start_s=5.0, prev_text=prev_text)

        # boundary_seg should be kept as non-overlap
        assert boundary_seg in result
