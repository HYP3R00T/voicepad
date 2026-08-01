import pytest
from voicepad_core.pipeline import AliasRule, ObservedWord, apply_aliases, ensure_terminal_punctuation


def word(text: str, start: int, end: int) -> ObservedWord:
    return ObservedWord(text, start, end, 0, 0, 1000)


def test_alias_correction_replaces_explicit_phrase_and_preserves_timing() -> None:
    words = (word("the", 0, 10), word("voice", 20, 30), word("pad,", 31, 40), word("works", 50, 60))
    rules = (AliasRule("VoicePad", ("voice pad",)),)

    result = apply_aliases(words, rules)

    assert result.text == "the VoicePad, works"
    assert result.words[1].start_sample == 20
    assert result.words[1].end_sample == 40
    assert result.corrections[0].observed == "voice pad,"


def test_alias_correction_does_not_fuzzily_rewrite_other_words() -> None:
    words = (word("voiced", 0, 10), word("padding", 20, 30))

    result = apply_aliases(words, (AliasRule("VoicePad", ("voice pad",)),))

    assert result.words == words
    assert result.corrections == ()


def test_complete_alphanumeric_text_receives_recorded_terminal_period() -> None:
    result = ensure_terminal_punctuation((word("finished", 0, 10),))

    assert result.text == "finished."
    assert result.corrections[0].canonical == "."


def test_existing_terminal_punctuation_is_preserved() -> None:
    words = (word("finished!", 0, 10),)

    assert ensure_terminal_punctuation(words).words == words


def test_alias_rules_reject_duplicate_or_empty_alternatives() -> None:
    with pytest.raises(ValueError, match="nonempty and unique"):
        AliasRule("VoicePad", ("voice pad", "Voice Pad"))
