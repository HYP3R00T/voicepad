from __future__ import annotations

import unicodedata

from voicepad_core.inference import BackendResult, TimedWord, TokenTimestamp
from voicepad_core.planning import AudioChunk
from voicepad_core.vad import SAMPLE_RATE, SpeechRegion

from .types import CoverageGap, ObservedToken, ObservedWord

DUPLICATE_TIME_TOLERANCE_SAMPLES = round(0.75 * SAMPLE_RATE)
COVERAGE_TIME_TOLERANCE_SAMPLES = round(0.5 * SAMPLE_RATE)
MAXIMUM_UNCOVERED_SPEECH_SAMPLES = 3 * SAMPLE_RATE


class ConservativeAssembler:
    """Collapse only timestamp-compatible duplicates and preserve uncertainty."""

    def __init__(self) -> None:
        self._words: list[ObservedWord] = []
        self._tokens: list[ObservedToken] = []
        self._warnings: list[str] = []
        self._protocol_valid = True

    @property
    def words(self) -> tuple[ObservedWord, ...]:
        return tuple(sorted(self._words, key=lambda word: (word.midpoint_sample, word.chunk_index)))

    @property
    def tokens(self) -> tuple[ObservedToken, ...]:
        return tuple(sorted(self._tokens, key=lambda token: (token.start_sample, token.chunk_index)))

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def protocol_valid(self) -> bool:
        return self._protocol_valid

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()

    def add(self, index: int, descriptor: AudioChunk, result: BackendResult) -> None:
        words = tuple(_absolute_word(index, descriptor, word) for word in result.words)
        rendered = " ".join(word.text for word in words).strip()
        if rendered != result.text.strip():
            self._protocol_valid = False
            self._warnings.append(f"chunk {index} timed words do not reproduce native decoded text")

        self._tokens.extend(_absolute_token(index, descriptor, token) for token in result.tokens)
        overlap_start = descriptor.source_start_sample
        overlap_end = descriptor.logical_start_sample
        for current in words:
            if overlap_start < overlap_end and current.midpoint_sample < overlap_end:
                match = self._duplicate(current, overlap_start, overlap_end)
                if match is not None:
                    if current.edge_distance_samples > match.edge_distance_samples:
                        self._words[self._words.index(match)] = current
                    continue
                self._warnings.append(
                    f"chunk {index} preserved unmatched overlap observation at sample {current.midpoint_sample}"
                )
            self._words.append(current)

    def coverage_gaps(self, speech_regions: tuple[SpeechRegion, ...]) -> tuple[CoverageGap, ...]:
        words = self.words
        gaps: list[CoverageGap] = []
        for speech in speech_regions:
            evidence = [
                word
                for word in words
                if word.end_sample >= speech.start_sample - COVERAGE_TIME_TOLERANCE_SAMPLES
                and word.start_sample <= speech.end_sample + COVERAGE_TIME_TOLERANCE_SAMPLES
            ]
            if not evidence:
                gaps.append(CoverageGap(speech, "no timed word plausibly covers VAD-confirmed speech"))
                continue
            cursor = speech.start_sample
            for word in evidence:
                if word.start_sample - cursor > MAXIMUM_UNCOVERED_SPEECH_SAMPLES:
                    gaps.append(
                        CoverageGap(
                            SpeechRegion(cursor, word.start_sample),
                            "timed-word gap exceeds the speech coverage tolerance",
                        )
                    )
                cursor = max(cursor, word.end_sample)
            if speech.end_sample - cursor > MAXIMUM_UNCOVERED_SPEECH_SAMPLES:
                gaps.append(
                    CoverageGap(
                        SpeechRegion(cursor, speech.end_sample),
                        "trailing timed-word gap exceeds the speech coverage tolerance",
                    )
                )
        return tuple(gaps)

    def _duplicate(self, current: ObservedWord, overlap_start: int, overlap_end: int) -> ObservedWord | None:
        normalized = _normalize_for_comparison(current.text)
        if not normalized:
            return None
        candidates = [
            previous
            for previous in self._words
            if previous.chunk_index != current.chunk_index
            and overlap_start <= previous.midpoint_sample <= overlap_end
            and _normalize_for_comparison(previous.text) == normalized
            and abs(previous.midpoint_sample - current.midpoint_sample) <= DUPLICATE_TIME_TOLERANCE_SAMPLES
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda word: abs(word.midpoint_sample - current.midpoint_sample))


def _absolute_word(index: int, descriptor: AudioChunk, word: TimedWord) -> ObservedWord:
    return ObservedWord(
        word.text,
        descriptor.source_start_sample + round(word.start_seconds * SAMPLE_RATE),
        descriptor.source_start_sample + round(word.end_seconds * SAMPLE_RATE),
        index,
        descriptor.source_start_sample,
        descriptor.source_end_sample,
    )


def _absolute_token(index: int, descriptor: AudioChunk, token: TokenTimestamp) -> ObservedToken:
    return ObservedToken(
        token.text,
        descriptor.source_start_sample + round(token.start_seconds * SAMPLE_RATE),
        descriptor.source_start_sample + round(token.end_seconds * SAMPLE_RATE),
        index,
    )


def _normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if not unicodedata.category(character).startswith("P"))
