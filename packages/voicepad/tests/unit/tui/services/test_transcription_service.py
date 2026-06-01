"""Tests for TranscriptionService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad.tui.services.transcription_service import TranscriptionService
from voicepad_core import TranscriptionError
from voicepad_core.inference import AudioTooShortError


def create_mock_config() -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.transcription_model = "turbo"
    config.transcription_device = "cuda"
    config.transcription_compute_type = "float16"
    config.logs_path = MagicMock()
    config.logs_path.mkdir = MagicMock()
    config.log_level = "INFO"
    config.language = "en"
    config.local_agreement_file = None
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

    @patch("voicepad_core.inference.model_manager._model_cache", {})
    @patch("voicepad.tui.services.transcription_service.model_downloaded")
    @patch("voicepad.tui.services.transcription_service.load_model")
    def test_warm_model_loads_model_when_cached(self, mock_load: MagicMock, mock_downloaded: MagicMock) -> None:
        """warm_model loads the model when it's already cached."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_downloaded.return_value = True
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = service.warm_model()

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_downloaded.assert_called_once_with("turbo")

    @patch("voicepad_core.inference.model_manager._model_cache", {})
    @patch("voicepad.tui.services.transcription_service.model_downloaded")
    @patch("voicepad.tui.services.transcription_service.ensure_model_downloaded")
    @patch("voicepad.tui.services.transcription_service.load_model")
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
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = service.warm_model()

        mock_ensure.assert_called_once_with("turbo")
        assert result.device == "cuda"

    @patch("voicepad.tui.services.transcription_service.model_downloaded")
    @patch("voicepad.tui.services.transcription_service.load_model")
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
    @patch("voicepad.tui.services.transcription_service.load_model")
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

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad.tui.services.transcription_service.transcribe")
    def test_transcribe_audio_returns_result(
        self, mock_transcribe: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock
    ) -> None:
        """transcribe_audio returns transcription result."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        audio = np.array([0.1, 0.2, 0.3])
        expected_result = create_mock_transcription_result()
        mock_transcribe.return_value = expected_result

        result = service.transcribe_audio(audio)

        assert result == expected_result

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad.tui.services.transcription_service.transcribe")
    def test_transcribe_audio_raises_on_audio_too_short(
        self, mock_transcribe: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock
    ) -> None:
        """transcribe_audio raises AudioTooShortError for short audio."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        audio = np.array([0.1, 0.2])
        mock_transcribe.side_effect = AudioTooShortError("Audio too short")

        with pytest.raises(AudioTooShortError, match="Audio too short"):
            service.transcribe_audio(audio)

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad.tui.services.transcription_service.transcribe")
    def test_transcribe_audio_raises_on_transcription_error(
        self, mock_transcribe: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock
    ) -> None:
        """transcribe_audio raises TranscriptionError on failure."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        audio = np.array([0.1, 0.2, 0.3])
        mock_transcribe.side_effect = TranscriptionError("Transcription failed")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            service.transcribe_audio(audio)

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad.tui.services.transcription_service.transcribe")
    def test_transcribe_audio_raises_on_unexpected_error(
        self, mock_transcribe: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock
    ) -> None:
        """transcribe_audio raises exception on unexpected error."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        audio = np.array([0.1, 0.2, 0.3])
        mock_transcribe.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(RuntimeError, match="Unexpected error"):
            service.transcribe_audio(audio)

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad_core.transcribe_file")
    def test_transcribe_file_loads_and_transcribes(
        self, mock_transcribe_file: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file loads audio file and transcribes it."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        expected_result = create_mock_transcription_result()
        mock_transcribe_file.return_value = expected_result

        result = service.transcribe_file(wav_path)

        assert result == expected_result
        mock_transcribe_file.assert_called_once()

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad_core.transcribe_file")
    def test_transcribe_file_raises_on_missing_file(
        self, mock_transcribe_file: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file raises FileNotFoundError for missing file."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        wav_path = tmp_path / "nonexistent.wav"
        mock_transcribe_file.side_effect = FileNotFoundError("Audio file not found")

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            service.transcribe_file(wav_path)

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad_core.transcribe_file")
    def test_transcribe_file_resamples_non_16khz_audio(
        self,
        mock_transcribe_file: MagicMock,
        mock_setup_logger: MagicMock,
        mock_set_logger: MagicMock,
        tmp_path: Path,
    ) -> None:
        """transcribe_file handles resampling via core_transcribe_file."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        expected_result = create_mock_transcription_result()
        mock_transcribe_file.return_value = expected_result

        result = service.transcribe_file(wav_path)

        assert result == expected_result
        mock_transcribe_file.assert_called_once()

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad_core.transcribe_file")
    def test_transcribe_file_raises_on_audio_too_short(
        self, mock_transcribe_file: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file raises AudioTooShortError for short audio."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        mock_transcribe_file.side_effect = AudioTooShortError("Audio too short")

        with pytest.raises(AudioTooShortError, match="Audio too short"):
            service.transcribe_file(wav_path)

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad_core.transcribe_file")
    def test_transcribe_file_raises_on_transcription_error(
        self, mock_transcribe_file: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file raises TranscriptionError on failure."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        mock_transcribe_file.side_effect = TranscriptionError("Transcription failed")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            service.transcribe_file(wav_path)

    @patch("voicepad.tui.services.transcription_service.set_session_logger")
    @patch("voicepad.tui.services.transcription_service.setup_transcription_logger")
    @patch("voicepad_core.transcribe_file")
    def test_transcribe_file_raises_on_read_error(
        self, mock_transcribe_file: MagicMock, mock_setup_logger: MagicMock, mock_set_logger: MagicMock, tmp_path: Path
    ) -> None:
        """transcribe_file raises exception on file read error."""
        config = create_mock_config()
        service = TranscriptionService(config)

        mock_logger = MagicMock()
        mock_setup_logger.return_value = (mock_logger, MagicMock())

        wav_path = tmp_path / "test.wav"
        wav_path.write_text("fake wav")

        mock_transcribe_file.side_effect = RuntimeError("Read failed")

        with pytest.raises(RuntimeError, match="Read failed"):
            service.transcribe_file(wav_path)
