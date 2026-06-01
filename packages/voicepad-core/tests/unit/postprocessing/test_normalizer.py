"""Tests for text normalization postprocessing."""

from voicepad_core.postprocessing.normalizer import normalize


class TestNormalize:
    """Test suite for normalize function."""

    def test_empty_text_returns_empty(self) -> None:
        """Empty text should return empty string."""
        assert normalize("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only text should return empty string."""
        assert normalize("   \t\n  ") == ""

    def test_single_word_unchanged(self) -> None:
        """Single word should remain unchanged."""
        assert normalize("hello") == "hello"

    def test_multiple_spaces_collapsed(self) -> None:
        """Multiple spaces should be collapsed to single space."""
        assert normalize("hello    world") == "hello world"

    def test_leading_trailing_spaces_removed(self) -> None:
        """Leading and trailing spaces should be removed."""
        assert normalize("  hello world  ") == "hello world"

    def test_preserves_punctuation(self) -> None:
        """Punctuation should be preserved."""
        assert normalize("Hello, world!") == "Hello, world!"

    def test_preserves_case(self) -> None:
        """Case should be preserved."""
        assert normalize("Hello World") == "Hello World"

    def test_mixed_content(self) -> None:
        """Mixed content with various whitespace should be normalized."""
        text = "  This  is   a   test  with   mixed  whitespace.  "
        expected = "This is a test with mixed whitespace."
        assert normalize(text) == expected

    def test_removes_blank_audio_artefact(self) -> None:
        """[BLANK_AUDIO] artefact should be removed."""
        assert normalize("Hello [BLANK_AUDIO] world") == "Hello world"

    def test_removes_music_artefact(self) -> None:
        """[MUSIC] artefact should be removed."""
        assert normalize("Hello [MUSIC] world") == "Hello world"

    def test_removes_noise_artefact(self) -> None:
        """[NOISE] artefact should be removed."""
        assert normalize("Hello [NOISE] world") == "Hello world"

    def test_removes_inaudible_artefact(self) -> None:
        """[INAUDIBLE] artefact should be removed."""
        assert normalize("Hello [INAUDIBLE] world") == "Hello world"

    def test_removes_parenthesis_music_artefact(self) -> None:
        """(MUSIC) artefact should be removed."""
        assert normalize("Hello (MUSIC) world") == "Hello world"

    def test_removes_parenthesis_noise_artefact(self) -> None:
        """(NOISE) artefact should be removed."""
        assert normalize("Hello (NOISE) world") == "Hello world"

    def test_removes_parenthesis_inaudible_artefact(self) -> None:
        """(INAUDIBLE) artefact should be removed."""
        assert normalize("Hello (INAUDIBLE) world") == "Hello world"

    def test_removes_asterisk_artefact(self) -> None:
        """*** artefact should be removed."""
        assert normalize("Hello *** world") == "Hello world"

    def test_removes_multiple_dots_artefact(self) -> None:
        """Four or more consecutive dots should be removed."""
        assert normalize("Hello .... world") == "Hello world"
        assert normalize("Hello ..... world") == "Hello world"

    def test_preserves_three_dots(self) -> None:
        """Three dots (ellipsis) should be preserved."""
        assert normalize("Hello... world") == "Hello... world"

    def test_artefact_removal_case_insensitive(self) -> None:
        """Artefact removal should be case insensitive."""
        assert normalize("Hello [blank_audio] world") == "Hello world"
        assert normalize("Hello [Music] world") == "Hello world"

    def test_multiple_artefacts_removed(self) -> None:
        """Multiple artefacts should all be removed."""
        text = "Hello [MUSIC] world [NOISE] test [BLANK_AUDIO] end"
        assert normalize(text) == "Hello world test end"

    def test_artefact_at_start(self) -> None:
        """Artefact at start should be removed."""
        assert normalize("[MUSIC] Hello world") == "Hello world"

    def test_artefact_at_end(self) -> None:
        """Artefact at end should be removed."""
        assert normalize("Hello world [MUSIC]") == "Hello world"

    def test_only_artefact_returns_empty(self) -> None:
        """Text with only artefacts should return empty string."""
        assert normalize("[MUSIC]") == ""
        assert normalize("[NOISE] [BLANK_AUDIO]") == ""

    def test_unicode_text_preserved(self) -> None:
        """Unicode characters should be preserved."""
        assert normalize("  Hello  世界  ") == "Hello 世界"

    def test_special_characters_preserved(self) -> None:
        """Special characters should be preserved."""
        assert normalize("  $100  @user  #tag  ") == "$100 @user #tag"

    def test_numbers_preserved(self) -> None:
        """Numbers should be preserved."""
        assert normalize("  123  456  ") == "123 456"

    def test_long_text_normalized(self) -> None:
        """Long text should be normalized correctly."""
        text = "  " + "  ".join(["word"] * 100) + "  "
        expected = " ".join(["word"] * 100)
        assert normalize(text) == expected
