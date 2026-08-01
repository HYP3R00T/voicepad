from voicepad_core.inference import BackendResult, TimedWord, TokenTimestamp
from voicepad_core.pipeline import ConservativeAssembler
from voicepad_core.planning import AudioChunk, OverlapKind
from voicepad_core.vad import SpeechRegion

SR = 16_000


def backend(text: str, words: tuple[TimedWord, ...]) -> BackendResult:
    tokens = tuple(TokenTimestamp(word.text, word.start_seconds, word.end_seconds) for word in words)
    return BackendResult(text, tokens, words)


def test_assembler_collapses_timestamp_compatible_overlap_duplicate() -> None:
    assembler = ConservativeAssembler()
    first = AudioChunk(0, 30 * SR, 0, 30 * SR, OverlapKind.NONE)
    second = AudioChunk(20 * SR, 55 * SR, 30 * SR, 55 * SR, OverlapKind.NATURAL)
    assembler.add(
        0,
        first,
        backend("before VoicePad", (TimedWord("before", 10, 11), TimedWord("VoicePad", 28, 28.5))),
    )
    assembler.add(
        1,
        second,
        backend("VoicePad continues", (TimedWord("VoicePad", 8.1, 8.6), TimedWord("continues", 11, 12))),
    )

    assert assembler.text == "before VoicePad continues"
    assert len(assembler.words) == 3
    assert assembler.words[1].chunk_index == 1
    assert assembler.warnings == ()


def test_assembler_preserves_uncertain_overlap_text_with_warning() -> None:
    assembler = ConservativeAssembler()
    first = AudioChunk(0, 30 * SR, 0, 30 * SR, OverlapKind.NONE)
    second = AudioChunk(20 * SR, 55 * SR, 30 * SR, 55 * SR, OverlapKind.NATURAL)
    assembler.add(0, first, backend("alpha", (TimedWord("alpha", 28, 28.5),)))
    assembler.add(1, second, backend("beta", (TimedWord("beta", 8.2, 8.7),)))

    assert assembler.text == "alpha beta"
    assert "preserved unmatched overlap" in assembler.warnings[0]


def test_assembler_marks_native_text_timestamp_mismatch() -> None:
    assembler = ConservativeAssembler()
    descriptor = AudioChunk(0, 10 * SR, 0, 10 * SR, OverlapKind.NONE, terminal=True)

    assembler.add(0, descriptor, backend("native text", (TimedWord("different", 1, 2),)))

    assert assembler.protocol_valid is False


def test_coverage_reports_speech_without_timed_words() -> None:
    assembler = ConservativeAssembler()
    descriptor = AudioChunk(0, 10 * SR, 0, 10 * SR, OverlapKind.NONE, terminal=True)
    assembler.add(0, descriptor, backend("covered", (TimedWord("covered", 1, 2),)))

    gaps = assembler.coverage_gaps((SpeechRegion(0, 3 * SR), SpeechRegion(7 * SR, 8 * SR)))

    assert len(gaps) == 1
    assert gaps[0].speech.start_sample == 7 * SR


def test_coverage_reports_long_internal_timestamp_gap() -> None:
    assembler = ConservativeAssembler()
    descriptor = AudioChunk(0, 12 * SR, 0, 12 * SR, OverlapKind.NONE, terminal=True)
    assembler.add(
        0,
        descriptor,
        backend("first last", (TimedWord("first", 1, 2), TimedWord("last", 9, 10))),
    )

    gaps = assembler.coverage_gaps((SpeechRegion(0, 11 * SR),))

    assert len(gaps) == 1
    assert gaps[0].reason == "timed-word gap exceeds the speech coverage tolerance"
