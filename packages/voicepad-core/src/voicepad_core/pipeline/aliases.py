from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from itertools import takewhile

from .types import AppliedCorrection, ObservedWord


@dataclass(frozen=True, slots=True)
class AliasRule:
    canonical: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.canonical.strip() or not self.aliases:
            raise ValueError("alias rule requires a canonical spelling and explicit alternatives")
        normalized = [_normalized_phrase(alias) for alias in self.aliases]
        if any(not alias for alias in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("alias alternatives must be nonempty and unique")


@dataclass(frozen=True, slots=True)
class AliasCorrectionResult:
    words: tuple[ObservedWord, ...]
    corrections: tuple[AppliedCorrection, ...]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()


def ensure_terminal_punctuation(words: tuple[ObservedWord, ...]) -> AliasCorrectionResult:
    if not words or not words[-1].text.rstrip() or not words[-1].text.rstrip()[-1].isalnum():
        return AliasCorrectionResult(words, ())
    last = words[-1]
    punctuated = ObservedWord(
        f"{last.text}.",
        last.start_sample,
        last.end_sample,
        last.chunk_index,
        last.physical_start_sample,
        last.physical_end_sample,
    )
    correction = AppliedCorrection(".", last.text, last.start_sample, last.end_sample)
    return AliasCorrectionResult((*words[:-1], punctuated), (correction,))


def apply_aliases(words: tuple[ObservedWord, ...], rules: tuple[AliasRule, ...]) -> AliasCorrectionResult:
    alternatives = sorted(
        ((tuple(_normalized_phrase(alias).split()), rule.canonical) for rule in rules for alias in rule.aliases),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    corrected: list[ObservedWord] = []
    applied: list[AppliedCorrection] = []
    index = 0
    while index < len(words):
        match = next(
            ((tokens, canonical) for tokens, canonical in alternatives if _matches(words, index, tokens)),
            None,
        )
        if match is None:
            corrected.append(words[index])
            index += 1
            continue
        tokens, canonical = match
        observed = words[index : index + len(tokens)]
        first = observed[0]
        last = observed[-1]
        replacement = ObservedWord(
            _preserve_outer_punctuation(first.text, last.text, canonical),
            first.start_sample,
            last.end_sample,
            first.chunk_index,
            min(word.physical_start_sample for word in observed),
            max(word.physical_end_sample for word in observed),
        )
        corrected.append(replacement)
        applied.append(
            AppliedCorrection(
                canonical,
                " ".join(word.text for word in observed),
                first.start_sample,
                last.end_sample,
            )
        )
        index += len(tokens)
    return AliasCorrectionResult(tuple(corrected), tuple(applied))


def _matches(words: tuple[ObservedWord, ...], index: int, tokens: tuple[str, ...]) -> bool:
    if index + len(tokens) > len(words):
        return False
    observed = tuple(_normalized_word(word.text) for word in words[index : index + len(tokens)])
    return observed == tokens


def _normalized_phrase(value: str) -> str:
    return " ".join(filter(None, (_normalized_word(token) for token in value.split())))


def _normalized_word(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not unicodedata.category(character).startswith("P"))


def _preserve_outer_punctuation(first: str, last: str, canonical: str) -> str:
    prefix = "".join(takewhile(_is_punctuation, first))
    suffix = "".join(reversed(tuple(takewhile(_is_punctuation, reversed(last)))))
    return f"{prefix}{canonical}{suffix}"


def _is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")
