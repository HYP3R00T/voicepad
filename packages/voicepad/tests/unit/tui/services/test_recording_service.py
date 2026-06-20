"""Tests for RecordingService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad.tui.services.recording_service import RecordingService


def create_mock_config(tmp_path: Path) -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.recordings_path = tmp_path / "recordings"
    return config


class TestRecordingService:
    """Test suite for RecordingService."""

    def test_init_stores_config(self, tmp_path: Path) -> None:
        """RecordingService stores the config object."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)
        assert service.config == config

    @patch("voicepad.tui.services.recording_service.RecordingSession")
    def test_create_session_returns_new_session(self, mock_session_class: MagicMock, tmp_path: Path) -> None:
        """create_session returns a new RecordingSession instance."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        session = service.create_session()

        mock_session_class.assert_called_once_with(config=config)
        assert session == mock_session

    def test_start_session_calls_session_start(self, tmp_path: Path) -> None:
        """start_session calls the session's start method."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        mock_session = MagicMock()
        service.start_session(mock_session)

        mock_session.start.assert_called_once()

    def test_start_session_raises_on_error(self, tmp_path: Path) -> None:
        """start_session raises RuntimeError if session fails to start."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        mock_session = MagicMock()
        mock_session.start.side_effect = RuntimeError("Failed to start")

        with pytest.raises(match="Failed to start"):
            service.start_session(mock_session)

    def test_stop_session_calls_session_stop(self, tmp_path: Path) -> None:
        """stop_session calls the session's stop method."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        mock_session = MagicMock()
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_session.stop.return_value = mock_audio

        audio = service.stop_session(mock_session)

        mock_session.stop.assert_called_once()
        np.testing.assert_array_equal(audio, mock_audio)

    def test_stop_session_raises_on_error(self, tmp_path: Path) -> None:
        """stop_session raises RuntimeError if session fails to stop."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        mock_session = MagicMock()
        mock_session.stop.side_effect = RuntimeError("Failed to stop")

        with pytest.raises(match="Failed to stop"):
            service.stop_session(mock_session)

    @patch("voicepad.tui.services.recording_service.sf.write")
    def test_save_audio_creates_directory(self, mock_write: MagicMock, tmp_path: Path) -> None:
        """save_audio creates recordings directory if it doesn't exist."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        audio = np.array([0.1, 0.2, 0.3])
        service.save_audio(audio, prefix="test")

        assert config.recordings_path.exists()

    @patch("voicepad.tui.services.recording_service.sf.write")
    def test_save_audio_writes_file(self, mock_write: MagicMock, tmp_path: Path) -> None:
        """save_audio writes audio to WAV file."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        audio = np.array([0.1, 0.2, 0.3])
        wav_path = service.save_audio(audio, prefix="test")

        expected_path = config.recordings_path / "test.wav"
        assert wav_path == expected_path
        mock_write.assert_called_once_with(expected_path, audio, 16000)

    @patch("voicepad.tui.services.recording_service.sf.write")
    @patch("voicepad.tui.services.recording_service.time.strftime")
    def test_save_audio_uses_timestamp_when_no_prefix(
        self, mock_strftime: MagicMock, mock_write: MagicMock, tmp_path: Path
    ) -> None:
        """save_audio uses timestamp as prefix when none provided."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        mock_strftime.return_value = "20240101_120000"
        audio = np.array([0.1, 0.2, 0.3])

        wav_path = service.save_audio(audio)

        expected_path = config.recordings_path / "20240101_120000.wav"
        assert wav_path == expected_path
        assert mock_strftime.call_args_list[0].args == ("%Y%m%d_%H%M%S",)

    @patch("voicepad.tui.services.recording_service.sf.write")
    def test_save_audio_raises_on_write_error(self, mock_write: MagicMock, tmp_path: Path) -> None:
        """save_audio raises exception if write fails."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        mock_write.side_effect = RuntimeError("Write failed")
        audio = np.array([0.1, 0.2, 0.3])

        with pytest.raises(Exception, match="Write failed"):
            service.save_audio(audio, prefix="test")

    def test_get_audio_duration_calculates_correctly(self, tmp_path: Path) -> None:
        """get_audio_duration calculates duration correctly."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        # 16000 samples = 1 second at 16kHz
        audio = np.zeros(16000)
        duration = service.get_audio_duration(audio)

        assert duration == 1.0

    def test_get_audio_duration_handles_fractional_seconds(self, tmp_path: Path) -> None:
        """get_audio_duration handles fractional seconds."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        # 8000 samples = 0.5 seconds at 16kHz
        audio = np.zeros(8000)
        duration = service.get_audio_duration(audio)

        assert duration == 0.5

    def test_get_audio_duration_handles_empty_audio(self, tmp_path: Path) -> None:
        """get_audio_duration handles empty audio array."""
        config = create_mock_config(tmp_path)
        service = RecordingService(config)

        audio = np.array([])
        duration = service.get_audio_duration(audio)

        assert duration == 0.0
