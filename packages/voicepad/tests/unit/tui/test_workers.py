from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from voicepad.tui.workers import (
    ModelWarmResult,
    RecordingSession,
    TranscriptionJob,
    warm_model,
)
from voicepad_core.audio import RawAudio, WavArtifact


def _audio(samples: list[float]) -> RawAudio:
    return RawAudio(np.array(samples, dtype=np.float32), 16_000, 1)


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

    @patch("voicepad_core.activate_model")
    @patch("voicepad_core.model_is_ready")
    def test_warm_model_when_already_downloaded(
        self,
        mock_model_is_ready: Mock,
        mock_activate: Mock,
    ) -> None:
        """A ready artifact is activated without preparing it again."""
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_config.transcription_device = "cuda"
        mock_config.transcription_compute_type = "float16"
        mock_model_is_ready.return_value = True
        mock_activate.return_value = MagicMock(
            device="cuda",
            precision="float16",
            fallback_to_cpu=False,
        )

        result = warm_model(mock_config)

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_model_is_ready.assert_called_once_with("base")
        mock_activate.assert_called_once_with("base", "cuda", "float16")

    @patch("voicepad_core.prepare_model")
    @patch("voicepad_core.activate_model")
    @patch("voicepad_core.model_is_ready")
    def test_warm_model_downloads_when_not_cached(
        self,
        mock_model_is_ready: Mock,
        mock_activate: Mock,
        mock_prepare: Mock,
    ) -> None:
        """A missing artifact is prepared before activation."""
        mock_config = MagicMock()
        mock_config.transcription_model = "large"
        mock_config.transcription_device = "cuda"
        mock_config.transcription_compute_type = "float16"
        mock_model_is_ready.return_value = False
        mock_activate.return_value = MagicMock(
            device="cuda",
            precision="float16",
            fallback_to_cpu=False,
        )

        result = warm_model(mock_config)

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        assert result.error is None
        mock_prepare.assert_called_once_with("large")

    @patch("voicepad_core.activate_model")
    @patch("voicepad_core.model_is_ready")
    def test_warm_model_logs_transcription_error_traceback(
        self,
        mock_model_is_ready: Mock,
        mock_activate: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A handled activation error is returned to the UI and logged with its traceback."""
        from voicepad_core import TranscriptionError

        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_is_ready.return_value = True
        mock_activate.side_effect = TranscriptionError("Model activation failed")

        result = warm_model(mock_config)

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Model activation failed"
        assert caplog.records[-1].exc_info is not None

    @patch("voicepad_core.activate_model")
    @patch("voicepad_core.model_is_ready")
    def test_warm_model_handles_generic_exception(
        self,
        mock_model_is_ready: Mock,
        mock_activate: Mock,
    ) -> None:
        """An unexpected activation failure is returned to the UI."""
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_model_is_ready.return_value = True
        mock_activate.side_effect = RuntimeError("Unexpected error")

        result = warm_model(mock_config)

        assert result.device == "cpu"
        assert result.compute_type == "int8"
        assert result.fallback is True
        assert result.error == "Unexpected error"

    @patch("voicepad_core.activate_model")
    @patch("voicepad_core.model_is_ready")
    def test_warm_model_with_fallback(
        self,
        mock_model_is_ready: Mock,
        mock_activate: Mock,
    ) -> None:
        """The actual CPU fallback reported by the runtime is preserved."""
        mock_config = MagicMock()
        mock_config.transcription_model = "base"
        mock_config.transcription_device = "cuda"
        mock_config.transcription_compute_type = "float16"
        mock_model_is_ready.return_value = True
        mock_activate.return_value = MagicMock(
            device="cpu",
            precision="int8",
            fallback_to_cpu=True,
        )

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
    def test_start_creates_and_starts_recorder(self, mock_recorder_class: Mock, tmp_path: Path) -> None:
        mock_config = MagicMock()
        mock_config.input_device_index = 0
        mock_recorder = MagicMock()
        mock_recorder_class.return_value = mock_recorder

        recording_path = tmp_path / "recording.wav"
        session = RecordingSession(config=mock_config, recording_path=recording_path)
        session.start()

        mock_recorder_class.assert_called_once_with(recording_path, device_index=0)
        mock_recorder.start.assert_called_once()

    @patch("voicepad.tui.workers.MicrophoneStream")
    def test_start_handles_audio_recorder_error(self, mock_recorder_class: Mock, tmp_path: Path) -> None:
        mock_config = MagicMock()
        mock_recorder = MagicMock()
        mock_recorder.start.side_effect = RuntimeError("Device busy")
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config, recording_path=tmp_path / "recording.wav")
        with pytest.raises(RuntimeError):
            session.start()

        assert session.error == "Device busy"

    @patch("voicepad.tui.workers.MicrophoneStream")
    def test_stop_returns_audio_from_recorder(
        self,
        mock_recorder_class: Mock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.input_device_index = 0
        mock_recorder = MagicMock()
        expected_audio = WavArtifact(Path("recording.wav"), 16_000, 1, 3, 3 / 16_000)
        mock_recorder.stop.return_value = expected_audio
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config, recording_path=Path("recording.wav"))
        session.start()
        audio = session.stop()

        assert audio == expected_audio
        mock_recorder.stop.assert_called_once()

    def test_stop_before_start_is_rejected(self) -> None:
        mock_config = MagicMock()
        session = RecordingSession(config=mock_config)

        with pytest.raises(RuntimeError, match="has not been started"):
            session.stop()

    @patch("voicepad.tui.workers.MicrophoneStream")
    def test_stop_handles_audio_recorder_error(self, mock_recorder_class: Mock) -> None:
        mock_config = MagicMock()
        mock_config.input_device_index = 0
        mock_recorder = MagicMock()
        mock_recorder.stop.side_effect = RuntimeError("Buffer overrun")
        mock_recorder_class.return_value = mock_recorder

        session = RecordingSession(config=mock_config, recording_path=Path("recording.wav"))
        session.start()

        with pytest.raises(RuntimeError):
            session.stop()

        assert session.error == "Buffer overrun"


class TestTranscriptionJob:
    """Test TranscriptionJob dataclass."""

    def test_init_stores_audio_and_config(self) -> None:
        audio = _audio([0.1, 0.2])
        mock_config = MagicMock()
        job = TranscriptionJob(audio=audio, config=mock_config)

        assert job.audio is audio
        assert job.config is mock_config

    def test_result_is_none_initially(self) -> None:
        audio = _audio([0.1, 0.2])
        mock_config = MagicMock()
        job = TranscriptionJob(audio=audio, config=mock_config)

        assert job.result is None

    def test_error_is_none_initially(self) -> None:
        audio = _audio([0.1, 0.2])
        mock_config = MagicMock()
        job = TranscriptionJob(audio=audio, config=mock_config)

        assert job.error is None

    @patch("voicepad_core.transcribe")
    def test_run_transcribes_audio_successfully(self, mock_transcribe: Mock) -> None:
        audio = _audio([0.1, 0.2])
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

        audio = _audio([0.1])
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

        audio = _audio([0.1, 0.2])
        mock_config = MagicMock()
        mock_transcribe.side_effect = TranscriptionError("Model failed")

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is None
        assert job.result is None
        assert job.error == "Model failed"

    @patch("voicepad_core.transcribe")
    def test_run_handles_generic_exception(self, mock_transcribe: Mock) -> None:
        audio = _audio([0.1, 0.2])
        mock_config = MagicMock()
        mock_transcribe.side_effect = RuntimeError("Unexpected error")

        job = TranscriptionJob(audio=audio, config=mock_config)
        result = job.run()

        assert result is None
        assert job.result is None
        assert job.error == "Unexpected error: Unexpected error"
