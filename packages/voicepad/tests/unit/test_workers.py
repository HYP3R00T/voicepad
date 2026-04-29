"""Tests for voicepad.tui.workers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad.tui.workers import ModelWarmResult, RecordingSession, TranscriptionJob, warm_model
from voicepad_core import AudioRecorderError
from voicepad_core.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Return a test config with temporary directories."""
    return Config(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
    )


# ---------------------------------------------------------------------------
# warm_model
# ---------------------------------------------------------------------------


class TestWarmModel:
    def test_warm_model_returns_result_on_success(self, config: Config) -> None:
        """When get_or_load_model succeeds, warm_model returns a ModelWarmResult."""
        mock_model = MagicMock()
        with patch("voicepad_core.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)):
            result = warm_model(config)

        assert isinstance(result, ModelWarmResult)
        assert result.device == "cuda"
        assert result.compute_type == "int8"
        assert result.fallback is False
        assert result.error is None

    def test_warm_model_returns_cpu_fallback_result(self, config: Config) -> None:
        """When get_or_load_model returns fallback=True, warm_model reflects that."""
        mock_model = MagicMock()
        with patch("voicepad_core.get_or_load_model", return_value=(mock_model, "cpu", "int8", True)):
            result = warm_model(config)

        assert result.device == "cpu"
        assert result.fallback is True
        assert result.error is None

    def test_warm_model_returns_error_result_on_exception(self, config: Config) -> None:
        """When get_or_load_model raises, warm_model returns a result with error set."""
        with patch("voicepad_core.get_or_load_model", side_effect=RuntimeError("boom")):
            result = warm_model(config)

        assert result.error == "boom"
        assert result.device == "cpu"
        assert result.fallback is True


# ---------------------------------------------------------------------------
# ModelWarmResult
# ---------------------------------------------------------------------------


class TestModelWarmResult:
    def test_model_warm_result_defaults_error_to_none(self) -> None:
        """ModelWarmResult.error defaults to None."""
        result = ModelWarmResult(device="cpu", compute_type="int8", fallback=False)
        assert result.error is None

    def test_model_warm_result_stores_all_fields(self) -> None:
        """ModelWarmResult stores device, compute_type, fallback, and error."""
        result = ModelWarmResult(device="cuda", compute_type="float16", fallback=True, error="oops")
        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is True
        assert result.error == "oops"


# ---------------------------------------------------------------------------
# RecordingSession
# ---------------------------------------------------------------------------


class TestRecordingSession:
    def test_start_opens_recorder(self, config: Config) -> None:
        """When start() is called, an AudioRecorder is created and started."""
        session = RecordingSession(config=config)
        mock_recorder = MagicMock()

        with patch("voicepad.tui.workers.AudioRecorder", return_value=mock_recorder):
            session.start()

        mock_recorder.start.assert_called_once()

    def test_start_sets_error_and_raises_on_audio_recorder_error(self, config: Config) -> None:
        """When AudioRecorder.start() raises AudioRecorderError, it propagates."""
        session = RecordingSession(config=config)
        mock_recorder = MagicMock()
        mock_recorder.start.side_effect = AudioRecorderError("device busy")

        with patch("voicepad.tui.workers.AudioRecorder", return_value=mock_recorder), pytest.raises(AudioRecorderError):
            session.start()

        assert session.error == "device busy"

    def test_stop_returns_empty_array_if_recorder_not_started(self, config: Config) -> None:
        """When stop() is called before start(), an empty array is returned."""
        session = RecordingSession(config=config)
        audio = session.stop()

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0

    def test_stop_returns_audio_from_recorder(self, config: Config) -> None:
        """When stop() is called after start(), the recorder's audio is returned."""
        session = RecordingSession(config=config)
        expected_audio = np.ones(16000, dtype=np.float32)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = expected_audio

        with patch("voicepad.tui.workers.AudioRecorder", return_value=mock_recorder):
            session.start()

        audio = session.stop()
        assert np.array_equal(audio, expected_audio)

    def test_stop_sets_error_and_raises_on_audio_recorder_error(self, config: Config) -> None:
        """When recorder.stop() raises AudioRecorderError, it propagates."""
        session = RecordingSession(config=config)
        mock_recorder = MagicMock()
        mock_recorder.stop.side_effect = AudioRecorderError("stop failed")

        with patch("voicepad.tui.workers.AudioRecorder", return_value=mock_recorder):
            session.start()

        with pytest.raises(AudioRecorderError):
            session.stop()

        assert session.error == "stop failed"

    def test_error_property_returns_none_initially(self, config: Config) -> None:
        """The error property is None before any failure."""
        session = RecordingSession(config=config)
        assert session.error is None


# ---------------------------------------------------------------------------
# TranscriptionJob
# ---------------------------------------------------------------------------


class TestTranscriptionJob:
    def test_run_returns_transcription_result_on_success(self, config: Config) -> None:
        """When transcribe_buffer succeeds, run() returns the result."""
        from voicepad_core.transcription import TranscriptionResult

        audio = np.zeros(16000, dtype=np.float32)
        mock_result = MagicMock(spec=TranscriptionResult)

        with patch("voicepad_core.transcribe_buffer", return_value=mock_result):
            job = TranscriptionJob(audio=audio, config=config)
            result = job.run()

        assert result is mock_result
        assert job.result is mock_result
        assert job.error is None

    def test_run_returns_none_and_sets_error_on_too_short(self, config: Config) -> None:
        """When AudioTooShortError is raised, run() returns None and sets a friendly error."""
        from voicepad_core.transcription import AudioTooShortError

        audio = np.zeros(100, dtype=np.float32)

        with patch("voicepad_core.transcribe_buffer", side_effect=AudioTooShortError("too short")):
            job = TranscriptionJob(audio=audio, config=config)
            result = job.run()

        assert result is None
        assert job.error is not None
        assert "0.5 seconds" in job.error

    def test_run_returns_none_and_sets_error_on_transcription_error(self, config: Config) -> None:
        """When TranscriptionError is raised, run() returns None and stores the error."""
        from voicepad_core import TranscriptionError

        audio = np.zeros(16000, dtype=np.float32)

        with patch("voicepad_core.transcribe_buffer", side_effect=TranscriptionError("model failed")):
            job = TranscriptionJob(audio=audio, config=config)
            result = job.run()

        assert result is None
        assert job.error == "model failed"

    def test_run_returns_none_and_sets_error_on_unexpected_exception(self, config: Config) -> None:
        """When an unexpected exception is raised, run() returns None and wraps the error."""
        audio = np.zeros(16000, dtype=np.float32)

        with patch("voicepad_core.transcribe_buffer", side_effect=ValueError("unexpected")):
            job = TranscriptionJob(audio=audio, config=config)
            result = job.run()

        assert result is None
        assert "unexpected" in (job.error or "")

    def test_run_initializes_result_and_error_to_none(self, config: Config) -> None:
        """Before run() is called, result and error are None."""
        audio = np.zeros(16000, dtype=np.float32)
        job = TranscriptionJob(audio=audio, config=config)

        assert job.result is None
        assert job.error is None
