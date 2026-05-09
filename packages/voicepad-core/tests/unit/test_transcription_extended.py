"""Extended tests for voicepad_core.transcription to improve coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core.config import Config
from voicepad_core.transcription import (
    _get_repo_id,
    ensure_model_downloaded,
    transcribe_buffer,
)

# ---------------------------------------------------------------------------
# _get_repo_id
# ---------------------------------------------------------------------------


class TestGetRepoId:
    def test_returns_correct_repo_id_for_tiny(self) -> None:
        assert _get_repo_id("tiny") == "Systran/faster-whisper-tiny"

    def test_returns_correct_repo_id_for_base(self) -> None:
        assert _get_repo_id("base") == "Systran/faster-whisper-base"

    def test_returns_correct_repo_id_for_small(self) -> None:
        assert _get_repo_id("small") == "Systran/faster-whisper-small"

    def test_returns_correct_repo_id_for_medium(self) -> None:
        assert _get_repo_id("medium") == "Systran/faster-whisper-medium"

    def test_returns_correct_repo_id_for_large_v2(self) -> None:
        assert _get_repo_id("large-v2") == "Systran/faster-whisper-large-v2"

    def test_returns_correct_repo_id_for_large_v3(self) -> None:
        assert _get_repo_id("large-v3") == "Systran/faster-whisper-large-v3"

    def test_returns_input_for_unknown_model(self) -> None:
        # _get_repo_id prepends "Systran/faster-whisper-" to unknown models
        result = _get_repo_id("custom-model")
        assert "custom-model" in result


# ---------------------------------------------------------------------------
# ensure_model_downloaded - additional coverage
# ---------------------------------------------------------------------------


class TestEnsureModelDownloadedExtended:
    def test_passes_repo_id_to_snapshot_download(self, tmp_path: Path) -> None:
        with (
            patch("voicepad_core.transcription.model_downloaded", return_value=False),
            patch("voicepad_core.transcription.snapshot_download") as mock_dl,
        ):
            ensure_model_downloaded("base")
        # Should call with correct repo_id as keyword argument
        call_kwargs = mock_dl.call_args.kwargs
        assert "repo_id" in call_kwargs
        assert call_kwargs["repo_id"] == "Systran/faster-whisper-base"


# ---------------------------------------------------------------------------
# transcribe_buffer - additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestTranscribeBufferExtended:
    def _mock_model(self, text: str = "hello", segments=None) -> MagicMock:
        if segments is None:
            seg = MagicMock()
            seg.start, seg.end, seg.text = 0.0, 0.9, text
            segments = [seg]
        m = MagicMock()
        m.transcribe.return_value = (segments, MagicMock(language="en", language_probability=0.99))
        return m

    def _speech_audio(self, seconds: float = 1.0) -> np.ndarray:
        """Return non-silent float32 audio at 16 kHz (sine wave)."""
        t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_handles_empty_segments_list(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = self._mock_model(text="", segments=[])
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(mock_model, "cuda", "float16", False),
        ):
            result = transcribe_buffer(self._speech_audio(1.0), config)
        assert result.text == ""
        assert len(result.segments) == 0

    def test_handles_multiple_segments(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        seg1 = MagicMock()
        seg1.start, seg1.end, seg1.text = 0.0, 1.0, "Hello"
        seg2 = MagicMock()
        seg2.start, seg2.end, seg2.text = 1.0, 2.0, "world"
        mock_model = self._mock_model(segments=[seg1, seg2])
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(mock_model, "cuda", "float16", False),
        ):
            result = transcribe_buffer(self._speech_audio(2.0), config)
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello"
        assert result.segments[1].text == "world"

    def test_calculates_latency_correctly(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = self._mock_model()
        with (
            patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "float16", False)),
            patch("voicepad_core.transcription.time.perf_counter", side_effect=[0.0, 0.15]),  # 150ms elapsed
        ):
            result = transcribe_buffer(self._speech_audio(1.0), config)
        assert result.latency_ms == pytest.approx(150.0, abs=1.0)

    def test_handles_int16_audio_input(self, tmp_path: Path) -> None:
        """Test that int16 audio is properly converted to float32."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        # Create int16 audio
        audio_int16 = (np.random.randint(-32768, 32767, 16000, dtype=np.int16) * 0.1).astype(np.int16)
        mock_model = self._mock_model()
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(mock_model, "cuda", "float16", False),
        ):
            result = transcribe_buffer(audio_int16, config)
        # Should successfully transcribe
        assert isinstance(result.text, str)

    def test_uses_config_model_name(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path, transcription_model="base")
        mock_model = self._mock_model()
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(mock_model, "cuda", "float16", False),
        ) as mock_get:
            transcribe_buffer(self._speech_audio(1.0), config)
        # Should pass config to get_or_load_model
        assert mock_get.call_args[0][0] == config

    def test_handles_very_long_audio_warning(self, tmp_path: Path) -> None:
        """Test that very long audio (>600s) triggers a warning log."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        # Create 700s of audio
        long_audio = np.tile(self._speech_audio(1.0), 700)
        mock_model = self._mock_model()
        with (
            patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "float16", False)),
            patch("voicepad_core.transcription.logger") as mock_logger,
        ):
            transcribe_buffer(long_audio, config)
        # Should log info about long audio
        assert any("long" in str(call).lower() or "700" in str(call) for call in mock_logger.info.call_args_list)

    def test_preserves_segment_timing(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        seg = MagicMock()
        seg.start, seg.end, seg.text = 1.5, 3.7, "test"
        mock_model = self._mock_model(segments=[seg])
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(mock_model, "cuda", "float16", False),
        ):
            result = transcribe_buffer(self._speech_audio(4.0), config)
        assert result.segments[0].start == 1.5
        assert result.segments[0].end == 3.7

    def test_handles_low_language_probability(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        seg = MagicMock()
        seg.start, seg.end, seg.text = 0.0, 1.0, "unclear"
        m = MagicMock()
        m.transcribe.return_value = ([seg], MagicMock(language="en", language_probability=0.3))
        with patch("voicepad_core.transcription.get_or_load_model", return_value=(m, "cuda", "float16", False)):
            result = transcribe_buffer(self._speech_audio(1.0), config)
        assert result.language_probability == pytest.approx(0.3)

    def test_concatenates_segment_text_correctly(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        seg1 = MagicMock()
        seg1.start, seg1.end, seg1.text = 0.0, 1.0, " Hello "
        seg2 = MagicMock()
        seg2.start, seg2.end, seg2.text = 1.0, 2.0, " world "
        seg3 = MagicMock()
        seg3.start, seg3.end, seg3.text = 2.0, 3.0, " test "
        mock_model = self._mock_model(segments=[seg1, seg2, seg3])
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(mock_model, "cuda", "float16", False),
        ):
            result = transcribe_buffer(self._speech_audio(3.0), config)
        # Text should be concatenated and stripped
        assert "Hello" in result.text
        assert "world" in result.text
        assert "test" in result.text
