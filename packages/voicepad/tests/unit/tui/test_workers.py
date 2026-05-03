"""Tests for workers.py module."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from voicepad.tui.workers import (
    ModelWarmResult,
    RecordingSession,
    TranscriptionJob,
    warm_model,
)


class TestModelWarmResult:
    """Test ModelWarmResult dataclass."""

    def test_init_with_all_fields(self) -> None:
        result = ModelWarmResult(
            device="cuda",
            compute_type="float16",
            fallback=False,
            error=None,
        )
        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None

    def test_init_with_error(self) -> None:
        result = ModelWarmResult(
            device="cpu",
            compute_type="int8",
            fallback=True,
            error="Model not found",
        )
        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Model not found"

    def test_default_error_is_none(self) -> None:
        result = ModelWarmResult(device="cuda", compute_type="float16", fallback=False)
        assert result.error is None


class TestWarmModel:
    """Test warm_model function."""

    @patch("voicepad_core.get_or_load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_when_already_downloaded(
        self,
        mock_model_downloaded: Mock,
        mock_get_or_load_model: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_downloaded.return_value = True
        mock_get_or_load_model.return_value = (None, "cuda", "float16", False)

        result = warm_model(mock_config)

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_model_downloaded.assert_called_once_with("base", mock_config)
        mock_get_or_load_model.assert_called_once_with(mock_config)

    @patch("voicepad_core.ensure_model_downloaded")
    @patch("voicepad_core.get_or_load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_downloads_when_not_cached(
        self,
        mock_model_downloaded: Mock,
        mock_get_or_load_model: Mock,
        mock_ensure_model_downloaded: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "large"
        mock_model_downloaded.return_value = False
        mock_get_or_load_model.return_value = (None, "cuda", "float16", False)

        result = warm_model(mock_config)

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_ensure_model_downloaded.assert_called_once_with("large", mock_config)

    @patch("voicepad_core.get_or_load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_handles_transcription_error(
        self,
        mock_model_downloaded: Mock,
        mock_get_or_load_model: Mock,
    ) -> None:
        from voicepad_core.transcription import TranscriptionError

        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_downloaded.return_value = True
        mock_get_or_load_model.side_effect = TranscriptionError("Model load failed")

        result = warm_model(mock_config)

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Model load failed"

    @patch("voicepad_core.get_or_load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_handles_generic_exception(
        self,
        mock_model_downloaded: Mock,
        mock_get_or_load_model: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_downloaded.return_value = True
        mock_get_or_load_model.side_effect = RuntimeError("Unexpected error")

        result = warm_model(mock_config)

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Unexpected error"

    @patch("voicepad_core.get_or_load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_with_fallback(
        self,
        mock_model_downloaded: Mock,
        mock_get_or_load_model: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_downloaded.return_value = True
        mock_get_or_load_model.return_value = (None, "cpu", "int8", True)

        result = warm_model(mock_config)

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error is None


class TestRecordingSession:
    """Test RecordingSession dataclass."""

    def test_init_stores_config(self) -> None:
        mock_config = MagicMock()
        session = RecordingSession(config=mock_config)
        assert session.config is mock_config

    def test_error_property_returns_none_initially(self) -> None:
        mock_config = MagicMock()
        session = RecordingSession(config=mock_config)
        assert session.error is None

    @patch("voicepad.tui.workers.AudioRecorder")
    def test_start_creates_and_starts_recorder(self, mock_recorder_class: Mock) -> None:
        mock_config = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config)
        session.start()

        mock_recorder_class.assert_called_once_with(mock_config)
        mock_recorder.start.assert_called_once()

    @patch("voicepad.tui.workers.AudioRecorder")
    def test_start_handles_audio_recorder_error(self, mock_recorder_class: Mock) -> None:
        from voicepad_core import AudioRecorderError

        mock_config = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.start.side_effect = AudioRecorderError("Mic not found")
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config)
        with pytest.raises(AudioRecorderError):
            session.start()

        assert session.error == "Mic not found"

    @patch("voicepad.tui.workers.AudioRecorder")
    def test_stop_returns_audio_from_recorder(self, mock_recorder_class: Mock) -> None:
        mock_config = MagicMock()
        mock_recorder = MagicMock()
        expected_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_recorder.stop.return_value = expected_audio
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config)
        session.start()
        audio = session.stop()

        np.testing.assert_array_equal(audio, expected_audio)
        mock_recorder.stop.assert_called_once()

    def test_stop_returns_empty_array_when_recorder_is_none(self) -> None:
        mock_config = MagicMock()
        session = RecordingSession(config=mock_config)
        audio = session.stop()

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 0

    @patch("voicepad.tui.workers.AudioRecorder")
    def test_stop_handles_audio_recorder_error(self, mock_recorder_class: Mock) -> None:
        from voicepad_core import AudioRecorderError

        mock_config = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.stop.side_effect = AudioRecorderError("Stop failed")
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config)
        session.start()

        with pytest.raises(AudioRecorderError):
            session.stop()

        assert session.error == "Stop failed"


class TestTranscriptionJob:
    """Test TranscriptionJob dataclass."""

    def test_init_stores_audio_and_config(self) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        job = TranscriptionJob(audio=audio, config=mock_config)

        np.testing.assert_array_equal(job.audio, audio)
        assert job.config is mock_config

    def test_result_is_none_initially(self) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        job = TranscriptionJob(audio=audio, config=mock_config)

        assert job.result is None

    def test_error_is_none_initially(self) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        job = TranscriptionJob(audio=audio, config=mock_config)

        assert job.error is None

    @patch("voicepad_core.transcribe_buffer")
    def test_run_transcribes_audio_successfully(self, mock_transcribe: Mock) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        mock_result = MagicMock()
        mock_transcribe.return_value = mock_result

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is mock_result
        assert job.result is mock_result
        assert job.error is None
        mock_transcribe.assert_called_once_with(audio, mock_config)

    @patch("voicepad_core.transcribe_buffer")
    def test_run_handles_audio_too_short_error(self, mock_transcribe: Mock) -> None:
        from voicepad_core.transcription import AudioTooShortError

        audio = np.array([0.1], dtype=np.float32)
        mock_config = MagicMock()
        mock_transcribe.side_effect = AudioTooShortError("Too short")

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is None
        assert job.result is None
        assert job.error == "Recording too short — speak for at least 0.5 seconds"

    @patch("voicepad_core.transcribe_buffer")
    def test_run_handles_transcription_error(self, mock_transcribe: Mock) -> None:
        from voicepad_core import TranscriptionError

        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        mock_transcribe.side_effect = TranscriptionError("Model failed")

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is None
        assert job.result is None
        assert job.error == "Model failed"

    @patch("voicepad_core.transcribe_buffer")
    def test_run_handles_generic_exception(self, mock_transcribe: Mock) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        mock_transcribe.side_effect = RuntimeError("Unexpected error")

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is None
        assert job.result is None
        assert job.error == "Unexpected error: Unexpected error"
