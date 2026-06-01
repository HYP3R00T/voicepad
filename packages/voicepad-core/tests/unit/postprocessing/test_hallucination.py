"""Tests for hallucination removal."""

from __future__ import annotations

from voicepad_core.postprocessing.hallucination import remove_hallucinations


class TestRemoveHallucinations:
    """Tests for remove_hallucinations function."""

    def test_empty_text_returns_empty(self) -> None:
        """Empty text should return empty."""
        assert remove_hallucinations("") == ""

    def test_short_text_unchanged(self) -> None:
        """Text shorter than max_repetitions+1 should be unchanged."""
        text = "hello world"
        assert remove_hallucinations(text, max_repetitions=3) == text

    def test_no_repetitions_unchanged(self) -> None:
        """Text without repetitions should be unchanged."""
        text = "the quick brown fox jumps over the lazy dog"
        assert remove_hallucinations(text) == text

    def test_single_word_repetition_within_limit(self) -> None:
        """Single word repeated within limit should be kept."""
        text = "the the cat"
        assert remove_hallucinations(text, max_repetitions=3) == text

    def test_single_word_repetition_exceeds_limit(self) -> None:
        """Single word repeated beyond limit should be trimmed."""
        text = "the the the the the cat"
        result = remove_hallucinations(text, max_repetitions=3)
        assert result == "the the the cat"

    def test_single_word_repetition_case_insensitive(self) -> None:
        """Single word repetition should be case-insensitive."""
        text = "The the THE the cat"
        result = remove_hallucinations(text, max_repetitions=2)
        # Preserves first occurrence's case for each kept repetition
        assert result == "The The cat"

    def test_multiple_single_word_repetitions(self) -> None:
        """Multiple different words with repetitions."""
        text = "the the the cat cat cat sat sat sat"
        result = remove_hallucinations(text, max_repetitions=2)
        assert result == "the the cat cat sat sat"

    def test_two_word_phrase_repetition(self) -> None:
        """Two-word phrase repeated 3+ times should be reduced to one."""
        text = "thank you thank you thank you for coming"
        result = remove_hallucinations(text)
        assert result == "thank you for coming"

    def test_two_word_phrase_repetition_case_insensitive(self) -> None:
        """Two-word phrase repetition should be case-insensitive."""
        text = "Thank you thank you THANK YOU for coming"
        result = remove_hallucinations(text)
        assert result == "Thank you for coming"

    def test_two_word_phrase_only_two_repetitions(self) -> None:
        """Two-word phrase repeated only twice should be kept."""
        text = "thank you thank you for coming"
        result = remove_hallucinations(text)
        assert result == text

    def test_two_word_phrase_many_repetitions(self) -> None:
        """Two-word phrase repeated many times should be reduced to one."""
        text = "thank you thank you thank you thank you thank you for coming"
        result = remove_hallucinations(text)
        assert result == "thank you for coming"

    def test_combined_single_and_phrase_repetitions(self) -> None:
        """Both single word and phrase repetitions."""
        text = "the the the the thank you thank you thank you cat"
        result = remove_hallucinations(text, max_repetitions=2)
        # First pass: "the the the the" -> "the the"
        # Second pass: "thank you" repeated 3x -> "thank you" (one copy)
        assert result == "the the thank you cat"

    def test_custom_max_repetitions(self) -> None:
        """Custom max_repetitions parameter."""
        text = "hello hello hello hello world"
        result = remove_hallucinations(text, max_repetitions=1)
        assert result == "hello world"

    def test_phrase_at_end_of_text(self) -> None:
        """Two-word phrase repetition at the end."""
        text = "hello world thank you thank you thank you"
        result = remove_hallucinations(text)
        assert result == "hello world thank you"

    def test_real_world_hallucination(self) -> None:
        """Real-world example of Whisper hallucination."""
        text = "Thank you for watching thank you for watching thank you for watching"
        result = remove_hallucinations(text)
        # This is a 3-word phrase, not 2-word, so it won't be caught by the 2-word phrase filter
        # The algorithm only handles 2-word phrases
        assert result == text  # Unchanged since it's a 3-word phrase

    def test_preserves_single_occurrence(self) -> None:
        """Single occurrence of words should be preserved."""
        text = "one two three four five"
        assert remove_hallucinations(text) == text

    def test_mixed_content_with_repetitions(self) -> None:
        """Mixed content with some repetitions."""
        text = "I said said said hello to the the cat"
        result = remove_hallucinations(text, max_repetitions=2)
        assert result == "I said said hello to the the cat"
