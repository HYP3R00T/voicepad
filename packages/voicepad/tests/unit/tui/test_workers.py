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

    @patch("voicepad_core._model_cache")
    @patch("voicepad_core.load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_when_already_downloaded(
        self,
        mock_model_downloaded: Mock,
        mock_load: Mock,
        mock_cache: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_config.transcription_device = "cuda"
        mock_config.transcription_compute_type = "float16"
        mock_model = MagicMock()
        mock_model_downloaded.return_value = True
        mock_load.return_value = mock_model

        # Mock the cache to return the device/compute info
        mock_cache.items.return_value = [(("base", "cuda", "float16"), mock_model)]

        result = warm_model(mock_config)

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_model_downloaded.assert_called_once_with("base")
        mock_load.assert_called_once_with("base", "cuda", "float16")

    @patch("voicepad_core._model_cache")
    @patch("voicepad_core.ensure_model_downloaded")
    @patch("voicepad_core.load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_downloads_when_not_cached(
        self,
        mock_model_downloaded: Mock,
        mock_load: Mock,
        mock_ensure_model_downloaded: Mock,
        mock_cache: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "large"
        mock_config.transcription_device = "cuda"
        mock_config.transcription_compute_type = "float16"
        mock_model = MagicMock()
        mock_model_downloaded.return_value = False
        mock_load.return_value = mock_model

        mock_cache.items.return_value = [(("large", "cuda", "float16"), mock_model)]

        result = warm_model(mock_config)

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_ensure_model_downloaded.assert_called_once_with("large")

    @patch("voicepad_core.load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_handles_transcription_error(
        self,
        mock_model_downloaded: Mock,
        mock_load: Mock,
    ) -> None:
        from voicepad_core import TranscriptionError

        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_downloaded.return_value = True
        mock_load.side_effect = TranscriptionError("Model load_model failed")

        result = warm_model(mock_config)

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Model load_model failed"

    @patch("voicepad_core.load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_handles_generic_exception(
        self,
        mock_model_downloaded: Mock,
        mock_load: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_downloaded.return_value = True
        mock_load.side_effect = RuntimeError("Unexpected error")

        result = warm_model(mock_config)

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Unexpected error"

    @patch("voicepad_core._model_cache")
    @patch("voicepad_core.load_model")
    @patch("voicepad_core.model_downloaded")
    def test_warm_model_with_fallback(
        self,
        mock_model_downloaded: Mock,
        mock_load: Mock,
        mock_cache: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_config.transcription_device = "cuda"  # Request CUDA
        mock_config.transcription_compute_type = "float16"
        mock_model = MagicMock()
        mock_model_downloaded.return_value = True
        mock_load.return_value = mock_model

        # Mock cache shows it loaded on CPU despite requesting CUDA (fallback)
        mock_cache.items.return_value = [(("base", "cpu", "int8"), mock_model)]

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

    @patch("voicepad.tui.workers.MicrophoneStream")
    def test_start_creates_and_starts_recorder(self, mock_recorder_class: Mock) -> None:
        mock_config = MagicMock()
        mock_config.input_device_index = 0
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config)
        session.start()

        mock_recorder_class.assert_called_once_with(0)
        mock_recorder.start.assert_called_once()

    @patch("voicepad.tui.workers.MicrophoneStream")
    def test_start_handles_audio_recorder_error(self, mock_recorder_class: Mock) -> None:
        mock_config = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.start.side_effect = RuntimeError("Device busy")
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config)
        with pytest.raises(RuntimeError):
            session.start()

        assert session.error == "Device busy"

    @patch("voicepad_core.AudioPreProcessor")
    @patch("voicepad.tui.workers.MicrophoneStream")
    def test_stop_returns_audio_from_recorder(
        self,
        mock_recorder_class: Mock,
        mock_preprocessor_class: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.input_device_index = 0
        mock_recorder = MagicMock()
        expected_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_recorder.stop.return_value = expected_audio
        mock_recorder.sample_rate = 16000
        mock_recorder_class.return_value = mock_recorder

        # Mock AudioPreProcessor
        mock_processor = MagicMock()
        mock_processor.process_array.return_value = expected_audio
        mock_preprocessor_class.return_value = mock_processor

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

    @patch("voicepad.tui.workers.MicrophoneStream")
    def test_stop_handles_audio_recorder_error(self, mock_recorder_class: Mock) -> None:
        mock_config = MagicMock()
        mock_config.input_device_index = 0
        mock_recorder = MagicMock()
        mock_recorder.stop.side_effect = RuntimeError("Buffer overrun")
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config)
        session.start()

        with pytest.raises(RuntimeError):
            session.stop()

        assert session.error == "Buffer overrun"


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

    @patch("voicepad_core.transcribe")
    def test_run_transcribes_audio_successfully(self, mock_transcribe: Mock) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_config.transcription_device = "cuda"
        mock_config.transcription_compute_type = "float16"
        mock_config.language = "en"
        mock_result = MagicMock()
        mock_transcribe.return_value = mock_result

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is mock_result
        assert job.result is mock_result
        assert job.error is None
        mock_transcribe.assert_called_once_with(
            audio,
            model_name="base",
            device="cuda",
            compute_type="float16",
            language="en",
            word_timestamps=False,
        )

    @patch("voicepad_core.transcribe")
    def test_run_handles_audio_too_short_error(self, mock_transcribe: Mock) -> None:
        from voicepad_core import AudioTooShortError

        audio = np.array([0.1], dtype=np.float32)
        mock_config = MagicMock()
        mock_transcribe.side_effect = AudioTooShortError("Too short")

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is None
        assert job.result is None
        assert job.error == "Recording too short — speak for at least 0.5 seconds"

    @patch("voicepad_core.transcribe")
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

    @patch("voicepad_core.transcribe")
    def test_run_handles_generic_exception(self, mock_transcribe: Mock) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        mock_config = MagicMock()
        mock_transcribe.side_effect = RuntimeError("Unexpected error")

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is None
        assert job.result is None
        assert job.error == "Unexpected error: Unexpected error"
