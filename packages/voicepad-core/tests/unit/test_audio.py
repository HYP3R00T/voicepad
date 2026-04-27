"""Tests for voicepad_core.audio."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core.audio import SAMPLE_RATE, AudioRecorder, AudioRecorderError
from voicepad_core.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Return a test config with temporary directories."""
    return Config(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
    )


class TestAudioRecorderInit:
    def test_init_creates_recordings_directory(self, tmp_path: Path) -> None:
        """When AudioRecorder is created, the recordings directory is created."""
        config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        AudioRecorder(config)
        assert (tmp_path / "recordings").exists()

    def test_init_sets_recording_false(self, config: Config) -> None:
        """When AudioRecorder is created, _recording is False."""
        recorder = AudioRecorder(config)
        assert recorder._recording is False

    def test_init_fails_if_recordings_dir_cannot_be_created(self) -> None:
        """When the recordings directory cannot be created, AudioRecorderError is raised."""
        # Use a path that's guaranteed to fail on all platforms
        import os

        invalid_path = Path(os.devnull) / "invalid" / "recordings"
        config = Config(
            recordings_path=invalid_path,
            markdown_path=Path(os.devnull),
        )
        with pytest.raises(AudioRecorderError):
            AudioRecorder(config)


class TestAudioRecorderStart:
    @patch("sounddevice.InputStream")
    def test_start_opens_stream_with_correct_parameters(self, mock_stream_class, config: Config) -> None:
        """When start() is called, a sounddevice stream is opened with correct settings."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()

        mock_stream_class.assert_called_once()
        call_kwargs = mock_stream_class.call_args[1]
        assert call_kwargs["device"] == config.input_device_index
        assert call_kwargs["channels"] == 1
        assert call_kwargs["samplerate"] == SAMPLE_RATE
        assert call_kwargs["dtype"] == "float32"

    @patch("sounddevice.InputStream")
    def test_start_calls_stream_start(self, mock_stream_class, config: Config) -> None:
        """When start() is called, stream.start() is called."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()

        mock_stream.start.assert_called_once()

    @patch("sounddevice.InputStream")
    def test_start_sets_recording_true(self, mock_stream_class, config: Config) -> None:
        """When start() succeeds, _recording is set to True."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()

        assert recorder._recording is True

    @patch("sounddevice.InputStream")
    def test_start_clears_frames(self, mock_stream_class, config: Config) -> None:
        """When start() is called, the frames buffer is cleared."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder._frames = [np.array([1.0, 2.0])]
        recorder.start()

        assert recorder._frames == []

    def test_start_raises_if_already_recording(self, config: Config) -> None:
        """When start() is called while already recording, AudioRecorderError is raised."""
        with patch("sounddevice.InputStream"):
            recorder = AudioRecorder(config)
            recorder.start()
            with pytest.raises(AudioRecorderError, match="Already recording"):
                recorder.start()

    @patch("sounddevice.InputStream")
    def test_start_raises_if_stream_cannot_open(self, mock_stream_class, config: Config) -> None:
        """When the stream cannot be opened, AudioRecorderError is raised."""
        mock_stream_class.side_effect = RuntimeError("Device not found")

        recorder = AudioRecorder(config)
        with pytest.raises(AudioRecorderError, match="Cannot open audio device"):
            recorder.start()


class TestAudioRecorderStop:
    @patch("sounddevice.InputStream")
    def test_stop_raises_if_not_recording(self, mock_stream_class, config: Config) -> None:
        """When stop() is called while not recording, AudioRecorderError is raised."""
        recorder = AudioRecorder(config)
        with pytest.raises(AudioRecorderError, match="Not recording"):
            recorder.stop()

    @patch("sounddevice.InputStream")
    def test_stop_sets_recording_false(self, mock_stream_class, config: Config) -> None:
        """When stop() is called, _recording is set to False."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()
        recorder.stop()

        assert recorder._recording is False

    @patch("sounddevice.InputStream")
    def test_stop_closes_stream(self, mock_stream_class, config: Config) -> None:
        """When stop() is called, stream.stop() and stream.close() are called."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()
        recorder.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("sounddevice.InputStream")
    def test_stop_returns_empty_array_if_no_frames(self, mock_stream_class, config: Config) -> None:
        """When stop() is called with no frames, an empty float32 array is returned."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()
        audio = recorder.stop()

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 0

    @patch("sounddevice.InputStream")
    def test_stop_returns_concatenated_frames(self, mock_stream_class, config: Config) -> None:
        """When stop() is called with frames, they are concatenated and returned."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()
        recorder._frames = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]
        audio = recorder.stop()

        assert audio.dtype == np.float32
        assert len(audio) == 4
        assert np.allclose(audio, [1.0, 2.0, 3.0, 4.0])

    @patch("sounddevice.InputStream")
    def test_stop_handles_stream_close_exception(self, mock_stream_class, config: Config) -> None:
        """When stream.close() raises an exception, stop() still returns audio."""
        mock_stream = MagicMock()
        mock_stream.close.side_effect = RuntimeError("Close failed")
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()
        recorder._frames = [np.array([1.0], dtype=np.float32)]
        audio = recorder.stop()

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1


class TestAudioRecorderIsRecording:
    @patch("sounddevice.InputStream")
    def test_is_recording_returns_false_initially(self, mock_stream_class, config: Config) -> None:
        """When is_recording() is called before start(), it returns False."""
        recorder = AudioRecorder(config)
        assert recorder.is_recording() is False

    @patch("sounddevice.InputStream")
    def test_is_recording_returns_true_after_start(self, mock_stream_class, config: Config) -> None:
        """When is_recording() is called after start(), it returns True."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()
        assert recorder.is_recording() is True

    @patch("sounddevice.InputStream")
    def test_is_recording_returns_false_after_stop(self, mock_stream_class, config: Config) -> None:
        """When is_recording() is called after stop(), it returns False."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        recorder = AudioRecorder(config)
        recorder.start()
        recorder.stop()
        assert recorder.is_recording() is False


class TestAudioRecorderSaveWav:
    def test_save_wav_creates_parent_directories(self, config: Config, tmp_path: Path) -> None:
        """When save_wav() is called, parent directories are created if needed."""
        recorder = AudioRecorder(config)
        nested_path = tmp_path / "a" / "b" / "c" / "audio.wav"
        audio = np.zeros(16000, dtype=np.float32)

        recorder.save_wav(audio, nested_path)

        assert nested_path.parent.exists()

    def test_save_wav_writes_wav_file(self, config: Config, tmp_path: Path) -> None:
        """When save_wav() is called, a WAV file is written."""
        recorder = AudioRecorder(config)
        wav_path = tmp_path / "test.wav"
        audio = np.ones(16000, dtype=np.float32) * 0.5

        recorder.save_wav(audio, wav_path)

        assert wav_path.exists()


class TestAudioRecorderMakeWavPath:
    def test_make_wav_path_uses_provided_prefix(self, config: Config) -> None:
        """When make_wav_path() is called with a prefix, it is used in the filename."""
        recorder = AudioRecorder(config)
        path = recorder.make_wav_path(prefix="meeting")
        assert "meeting" in path.name

    def test_make_wav_path_uses_config_prefix_when_none(self, config: Config) -> None:
        """When make_wav_path() is called without a prefix, config.recording_prefix is used."""
        recorder = AudioRecorder(config)
        path = recorder.make_wav_path(prefix=None)
        assert config.recording_prefix in path.name

    def test_make_wav_path_includes_timestamp(self, config: Config) -> None:
        """The returned path includes a timestamp in the format YYYYMMDD_HHMMSS."""
        recorder = AudioRecorder(config)
        path = recorder.make_wav_path()
        # Check that the filename has a timestamp pattern (8 digits + underscore + 6 digits)
        assert "_" in path.name
        parts = path.name.split("_")
        assert len(parts) >= 2

    def test_make_wav_path_returns_wav_extension(self, config: Config) -> None:
        """The returned path has a .wav extension."""
        recorder = AudioRecorder(config)
        path = recorder.make_wav_path()
        assert path.suffix == ".wav"

    def test_make_wav_path_is_under_recordings_path(self, config: Config) -> None:
        """The returned path is under config.recordings_path."""
        recorder = AudioRecorder(config)
        path = recorder.make_wav_path()
        assert path.parent == config.recordings_path
