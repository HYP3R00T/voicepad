"""Tests for Sherpa-backed Silero VAD."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core.vad.silero import SileroVAD, SpeechSegment


class _Detector:
    def __init__(self, segments: list[object] | None = None) -> None:
        self.segments = list(segments or [])
        self.accepted: list[np.ndarray] = []
        self.flush_calls = 0
        self.reset_calls = 0

    def accept_waveform(self, samples: np.ndarray) -> None:
        self.accepted.append(samples)

    def flush(self) -> None:
        self.flush_calls += 1

    def empty(self) -> bool:
        return not self.segments

    @property
    def front(self) -> object:
        return self.segments[0]

    def pop(self) -> None:
        self.segments.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


class TestSileroVAD:
    @patch("voicepad_core.vad.silero.SileroVAD._load_detector")
    def test_initialization(self, load_detector: MagicMock) -> None:
        detector = _Detector()
        load_detector.return_value = detector

        vad = SileroVAD(
            threshold=0.6,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
        )

        assert vad._detector is detector
        load_detector.assert_called_once_with(0.6, 250, 100)

    @patch("voicepad_core.vad.silero.SileroVAD._load_detector")
    def test_detect_rejects_wrong_sample_rate(self, load_detector: MagicMock) -> None:
        load_detector.return_value = _Detector()
        vad = SileroVAD()

        with pytest.raises(ValueError, match="SileroVAD requires audio at 16000Hz"):
            vad.detect(np.zeros(16_000, dtype=np.float32), sample_rate=48_000)

    @patch("voicepad_core.vad.silero.SileroVAD._load_detector")
    def test_detect_ignores_audio_shorter_than_one_window(self, load_detector: MagicMock) -> None:
        load_detector.return_value = _Detector()

        assert SileroVAD().detect(np.zeros(500, dtype=np.float32)) == []

    @patch("voicepad_core.vad.silero.SileroVAD._load_detector")
    def test_detect_feeds_windows_and_returns_padded_segments(self, load_detector: MagicMock) -> None:
        detector = _Detector([
            SimpleNamespace(start=512, samples=np.zeros(1536, dtype=np.float32)),
            SimpleNamespace(start=3584, samples=np.zeros(1024, dtype=np.float32)),
        ])
        load_detector.return_value = detector
        vad = SileroVAD(speech_pad_ms=10)

        segments = vad.detect(np.zeros(11 * 512, dtype=np.float64))

        assert len(detector.accepted) == 11
        assert all(window.shape == (512,) and window.dtype == np.float32 for window in detector.accepted)
        assert detector.flush_calls == 1
        assert segments == [
            SpeechSegment(start=352 / 16_000, end=2208 / 16_000),
            SpeechSegment(start=3424 / 16_000, end=4768 / 16_000),
        ]

    @patch("voicepad_core.vad.silero.SileroVAD._load_detector")
    def test_detect_pads_partial_window_without_extending_segment_boundary(
        self,
        load_detector: MagicMock,
    ) -> None:
        detector = _Detector([SimpleNamespace(start=512, samples=np.zeros(512, dtype=np.float32))])
        load_detector.return_value = detector
        vad = SileroVAD(speech_pad_ms=30)

        segments = vad.detect(np.zeros(600, dtype=np.float32))

        assert len(detector.accepted) == 2
        assert segments[0].end == 600 / 16_000

    @patch("voicepad_core.vad.silero.SileroVAD._load_detector")
    def test_reset_delegates_to_sherpa(self, load_detector: MagicMock) -> None:
        detector = _Detector()
        load_detector.return_value = detector
        vad = SileroVAD()

        vad.reset()

        assert detector.reset_calls == 1
