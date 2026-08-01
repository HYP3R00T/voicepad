import numpy as np
import pytest
from voicepad_core.inference import InferenceError, TimedWord, TokenTimestamp
from voicepad_core.inference.parakeet import _split_decoded, _timestamp_tokens, _tokens_to_words


def test_timestamp_records_flatten_and_validate() -> None:
    records = [
        [
            {"token": "▁Voice", "start": 0.1, "end": 0.3},
            {"token": "Pad", "start": 0.3, "end": 0.5},
            {"token": "▁works", "start": 0.6, "end": 0.9},
            {"token": ".", "start": 0.9, "end": 1.0},
        ]
    ]

    tokens = _timestamp_tokens(records)

    assert tokens == (
        TokenTimestamp("▁Voice", 0.1, 0.3),
        TokenTimestamp("Pad", 0.3, 0.5),
        TokenTimestamp("▁works", 0.6, 0.9),
        TokenTimestamp(".", 0.9, 1.0),
    )
    assert _tokens_to_words(tokens) == (
        TimedWord("VoicePad", 0.1, 0.5),
        TimedWord("works.", 0.6, 1.0),
    )


def test_timestamp_records_reject_invalid_duration() -> None:
    with pytest.raises(InferenceError, match="non-monotonic"):
        _timestamp_tokens([{"token": "bad", "start": np.nan, "end": 1.0}])


def test_decode_requires_text_and_timestamps() -> None:
    assert _split_decoded((["hello"], [])) == ("hello", [])

    with pytest.raises(InferenceError, match="text and token timestamps"):
        _split_decoded("hello")
