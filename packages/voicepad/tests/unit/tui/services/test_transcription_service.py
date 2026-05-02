"""Tests for TranscriptionService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad.tui.services.transcription_service import TranscriptionService
from voicepad_core import TranscriptionError
from voicepad_core.transcription import AudioTooShortError


def create_mock_config() -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.transcription_model = "turbo"
    return config


def create_mock_transcription_result() -> MagicMock:
    """Create a mock TranscriptionResult."""
    result = MagicMock()
    result.text = "Test transcription"
    result.latency_ms = 100.0
    result.device = "cpu"
    return result


class TestTranscriptionService:
    """Test suite for TranscriptionService."""

    def test_init_stores_config(self) -> None:
        """TranscriptionService stores the config object."""
        config = create_mock_config()
        service = TranscriptionService(config)
        assert service.config == config

    @patch("voicepad.tui.services.transcription_service.model_downloaded")
    @patch("voicepad.tui.services.transcription_service.get_or_load_model")
    def test_warm_model_loads_model_when_cached(self, mock_load: MagicMock, mock_downloaded: MagicMock) -> None:
        """warm_model loads the model when it's already cached."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_downloaded.return_value = True
        mock_load.return_value = (MagicMock(), "cuda", "float16", False)

        result = service.warm_model()

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_downloaded.assert_called_once_with("turbo", config)

    @patch("voicepad.tui.services.transcription_service.model_downloaded")
    @patch("voicepad.tui.services.transcription_service.ensure_model_downloaded")
    @patch("voicepad.tui.services.transcription_service.get_or_load_model")
    def test_warm_model_downloads_when_not_cached(
        self,
        mock_load: MagicMock,
        mock_ensure: MagicMock,
        mock_downloaded: MagicMock,
    ) -> None:
        """warm_model downloads the model when not cached."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_downloaded.return_value = False
        mock_load.return_value = (MagicMock(), "cuda", "float16", False)

        result = service.warm_model()

        mock_ensure.assert_called_once_with("turbo", config)
        assert result.device == "cuda"

    @patch("voicepad.tui.services.transcription_service.model_downloaded")
    @patch("voicepad.tui.services.transcription_service.get_or_load_model")
    def test_warm_model_returns_fallback_on_download_error(
        self, mock_load: MagicMock, mock_downloaded: MagicMock
    ) -> None:
        """warm_model returns fallback result on download error."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_downloaded.return_value = True
        mock_load.side_effect = TranscriptionError("Download failed")

        result = service.warm_model()

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Download failed"

    @patch("voicepad.tui.services.transcription_service.model_downloaded")
    @patch("voicepad.tui.services.transcription_service.get_or_load_model")
    def test_warm_model_returns_fallback_on_unexpected_error(
        self, mock_load: MagicMock, mock_downloaded: MagicMock
    ) -> None:
        """warm_model returns fallback result on unexpected error."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_downloaded.return_value = True
        mock_load.side_effect = RuntimeError("Unexpected error")

        result = service.warm_model()

        assert result.device == "cpu"
        assert result.fallback is True
        assert result.error == "Unexpected error"

    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    def test_transcribe_audio_returns_result(self, mock_transcribe: MagicMock) -> None:
        """transcribe_audio returns transcription result."""
        config = create_mock_config()
        service = TranscriptionService(config)

        audio = np.array([0.1, 0.2, 0.3])
        expected_result = create_mock_transcription_result()
        mock_transcribe.return_value = expected_result

        result = service.transcribe_audio(audio)

        assert result == expected_result
        mock_transcribe.assert_called_once_with(audio, config)

    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    def test_transcribe_audio_raises_on_audio_too_short(self, mock_transcribe: MagicMock) -> None:
        """transcribe_audio raises AudioTooShortError for short audio."""
        config = create_mock_config()
        service = TranscriptionService(config)

        audio = np.array([0.1, 0.2])
        mock_transcribe.side_effect = AudioTooShortError("Audio too short")

        with pytest.raises(AudioTooShortError, match="Audio too short"):
            service.transcribe_audio(audio)

    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    def test_transcribe_audio_raises_on_transcription_error(self, mock_transcribe: MagicMock) -> None:
        """transcribe_audio raises TranscriptionError on failure."""
        config = create_mock_config()
        service = TranscriptionService(config)

        audio = np.array([0.1, 0.2, 0.3])
        mock_transcribe.side_effect = TranscriptionError("Transcription failed")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            service.transcribe_audio(audio)

    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    def test_transcribe_audio_raises_on_unexpected_error(self, mock_transcribe: MagicMock) -> None:
        """transcribe_audio raises exception on unexpected error."""
        config = create_mock_config()
        service = TranscriptionService(config)

        audio = np.array([0.1, 0.2, 0.3])
        mock_transcribe.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(RuntimeError, match="Unexpected error"):
            service.transcribe_audio(audio)

    @patch("voicepad.tui.services.transcription_service.sf.read")
    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    def test_transcribe_file_loads_and_transcribes(
        self, mock_transcribe: MagicMock, mock_read: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file loads audio file and transcribes it."""
        config = create_mock_config()
        service = TranscriptionService(config)

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        audio = np.array([0.1, 0.2, 0.3])
        mock_read.return_value = (audio, 16000)

        expected_result = create_mock_transcription_result()
        mock_transcribe.return_value = expected_result

        result = service.transcribe_file(wav_path)

        assert result == expected_result
        mock_read.assert_called_once_with(wav_path, dtype="float32")
        mock_transcribe.assert_called_once()

    def test_transcribe_file_raises_on_missing_file(self, tmp_path: Path) -> None:
        """transcribe_file raises FileNotFoundError for missing file."""
        config = create_mock_config()
        service = TranscriptionService(config)

        wav_path = tmp_path / "nonexistent.wav"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            service.transcribe_file(wav_path)

    @patch("voicepad.tui.services.transcription_service.sf.read")
    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    @patch("voicepad.tui.services.transcription_service.np.interp")
    def test_transcribe_file_resamples_non_16khz_audio(
        self,
        mock_interp: MagicMock,
        mock_transcribe: MagicMock,
        mock_read: MagicMock,
        tmp_path: Path,
    ) -> None:
        """transcribe_file resamples audio that's not 16kHz."""
        config = create_mock_config()
        service = TranscriptionService(config)

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        # Audio at 44.1kHz
        audio = np.array([0.1, 0.2, 0.3, 0.4])
        mock_read.return_value = (audio, 44100)

        resampled_audio = np.array([0.1, 0.2])
        mock_interp.return_value = resampled_audio

        expected_result = create_mock_transcription_result()
        mock_transcribe.return_value = expected_result

        result = service.transcribe_file(wav_path)

        assert result == expected_result
        mock_interp.assert_called_once()
        # Verify transcribe was called with resampled audio
        call_args = mock_transcribe.call_args[0]
        np.testing.assert_array_equal(call_args[0], resampled_audio)

    @patch("voicepad.tui.services.transcription_service.sf.read")
    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    def test_transcribe_file_raises_on_audio_too_short(
        self, mock_transcribe: MagicMock, mock_read: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file raises AudioTooShortError for short audio."""
        config = create_mock_config()
        service = TranscriptionService(config)

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        audio = np.array([0.1, 0.2])
        mock_read.return_value = (audio, 16000)
        mock_transcribe.side_effect = AudioTooShortError("Audio too short")

        with pytest.raises(AudioTooShortError, match="Audio too short"):
            service.transcribe_file(wav_path)

    @patch("voicepad.tui.services.transcription_service.sf.read")
    @patch("voicepad.tui.services.transcription_service.transcribe_buffer")
    def test_transcribe_file_raises_on_transcription_error(
        self, mock_transcribe: MagicMock, mock_read: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file raises TranscriptionError on failure."""
        config = create_mock_config()
        service = TranscriptionService(config)

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        audio = np.array([0.1, 0.2, 0.3])
        mock_read.return_value = (audio, 16000)
        mock_transcribe.side_effect = TranscriptionError("Transcription failed")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            service.transcribe_file(wav_path)

    @patch("voicepad.tui.services.transcription_service.sf.read")
    def test_transcribe_file_raises_on_read_error(self, mock_read: MagicMock, tmp_path: Path) -> None:
        """transcribe_file raises exception on file read error."""
        config = create_mock_config()
        service = TranscriptionService(config)

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        mock_read.side_effect = RuntimeError("Read failed")

        with pytest.raises(RuntimeError, match="Read failed"):
            service.transcribe_file(wav_path)
