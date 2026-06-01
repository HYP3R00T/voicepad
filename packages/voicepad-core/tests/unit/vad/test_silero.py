"""Tests for silero.py - SileroVAD class."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core.vad.silero import SileroVAD


class TestSileroVAD:
    """Test SileroVAD initialization and speech detection."""

    @patch("voicepad_core.vad.silero.SileroVAD._load_session")
    def test_initialization(self, mock_load_session: MagicMock) -> None:
        """Test SileroVAD initializes state correctly."""
        mock_session = MagicMock()
        mock_load_session.return_value = mock_session

        vad = SileroVAD(
            threshold=0.6,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
        )

        assert vad._threshold == 0.6
        assert vad._session == mock_session
        assert vad._h.shape == (1, 1, 128)
        assert vad._c.shape == (1, 1, 128)
        assert np.all(vad._h == 0)
        assert np.all(vad._c == 0)

    @patch("voicepad_core.vad.silero.SileroVAD._load_session")
    def test_detect_invalid_sample_rate(self, mock_load_session: MagicMock) -> None:
        """Test detect raises ValueError if sample rate is not 16000."""
        mock_load_session.return_value = MagicMock()
        vad = SileroVAD()

        with pytest.raises(ValueError, match="SileroVAD requires audio at 16000Hz"):
            vad.detect(np.zeros(16000, dtype=np.float32), sample_rate=48000)

    @patch("voicepad_core.vad.silero.SileroVAD._load_session")
    def test_detect_too_short(self, mock_load_session: MagicMock) -> None:
        """Test detect returns empty list if audio is too short (< 512 samples)."""
        mock_load_session.return_value = MagicMock()
        vad = SileroVAD()

        segments = vad.detect(np.zeros(500, dtype=np.float32))
        assert segments == []

    @patch("voicepad_core.vad.silero.SileroVAD._load_session")
    def test_detect_speech(self, mock_load_session: MagicMock) -> None:
        """Test detect processes audio and returns speech segments."""
        mock_session = MagicMock()
        # Mock InferenceSession.run to return:
        # - outputs[0]: speech probability (probs array)
        # - outputs[1]: hn
        # - outputs[2]: cn
        probs = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1], dtype=np.float32)
        hn = np.ones((1, 1, 128), dtype=np.float32) * 0.5
        cn = np.ones((1, 1, 128), dtype=np.float32) * 0.5
        mock_session.run.return_value = [probs, hn, cn]
        mock_load_session.return_value = mock_session

        # Minimum speech samples = 50ms * 16 = 800 samples.
        # Window size is 512 samples.
        # Probs length 11 -> audio length 11 * 512 = 5632 samples.
        vad = SileroVAD(
            threshold=0.5,
            min_speech_duration_ms=50,
            min_silence_duration_ms=50,
            speech_pad_ms=10,
        )

        audio = np.zeros(11 * 512, dtype=np.float32)
        segments = vad.detect(audio)

        # Let's inspect mock_session.run arguments
        mock_session.run.assert_called_once()
        args, _ = mock_session.run.call_args
        inputs = args[1]
        assert "input" in inputs
        assert inputs["input"].shape == (11, 576)  # shape (num_chunks, 512 + 64)
        assert np.allclose(vad._h, hn)
        assert np.allclose(vad._c, cn)
        assert len(segments) == 2

    @patch("voicepad_core.vad.silero.SileroVAD._load_session")
    def test_reset(self, mock_load_session: MagicMock) -> None:
        """Test reset restores LSTM states to zeros."""
        mock_load_session.return_value = MagicMock()
        vad = SileroVAD()

        vad._h = np.ones((1, 1, 128), dtype=np.float32)
        vad._c = np.ones((1, 1, 128), dtype=np.float32)

        vad.reset()

        assert np.all(vad._h == 0)
        assert np.all(vad._c == 0)
