"""Tests for microphone.py - MicrophoneStream class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from voicepad_core.audio.microphone import MicrophoneStream

# ============================================================================
# MicrophoneStream Tests
# ============================================================================# ============================================================================
# MicrophoneStream Tests
# ============================================================================


class TestMicrophoneStreamInitialization:
    """Test MicrophoneStream initialization."""

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_init_default_device(self, mock_query: Mock) -> None:
        """Test initialization with default device."""
        mock_query.return_value = {"default_samplerate": 48000}

        stream = MicrophoneStream()

        mock_query.assert_called_once_with(None, kind="input")
        assert stream.sample_rate == 48000
        assert stream._channels == 1
        assert stream._device_index is None
        assert not stream.is_recording

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_init_specific_device(self, mock_query: Mock) -> None:
        """Test initialization with specific device index."""
        mock_query.return_value = {"default_samplerate": 44100}

        stream = MicrophoneStream(device_index=2)

        mock_query.assert_called_once_with(2, kind="input")
        assert stream.sample_rate == 44100
        assert stream._device_index == 2

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_query_native_rate_fallback(self, mock_query: Mock) -> None:
        """Test fallback to 16000 Hz on query failure."""
        mock_query.side_effect = Exception("Device error")

        stream = MicrophoneStream()

        assert stream.sample_rate == 16000

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_query_native_rate_zero_fallback(self, mock_query: Mock) -> None:
        """Test fallback when rate is zero."""
        mock_query.return_value = {"default_samplerate": 0}

        stream = MicrophoneStream()

        assert stream.sample_rate == 16000

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_query_native_rate_negative_fallback(self, mock_query: Mock) -> None:
        """Test fallback when rate is negative."""
        mock_query.return_value = {"default_samplerate": -100}

        stream = MicrophoneStream()

        assert stream.sample_rate == 16000


class TestMicrophoneStreamRecording:
    """Test MicrophoneStream recording functionality."""

    @patch("voicepad_core.audio.microphone.sd.InputStream")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_start_opens_stream(self, mock_query: Mock, mock_stream_class: Mock) -> None:
        """Test start() opens the audio stream."""
        mock_query.return_value = {"default_samplerate": 16000}
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        stream = MicrophoneStream()
        stream.start()

        # Verify InputStream was created with correct parameters
        mock_stream_class.assert_called_once_with(
            samplerate=16000,
            channels=1,
            dtype="float32",
            device=None,
            callback=stream._callback,
        )
        mock_stream.start.assert_called_once()
        assert stream.is_recording

    @patch("voicepad_core.audio.microphone.sd.InputStream")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_start_clears_previous_frames(self, mock_query: Mock, mock_stream_class: Mock) -> None:
        """Test start() clears any previous frames."""
        mock_query.return_value = {"default_samplerate": 16000}
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        stream = MicrophoneStream()
        # Add some frames
        stream._frames.append(np.array([0.1, 0.2]))

        stream.start()

        # Frames should be cleared
        assert len(stream._frames) == 0

    @patch("voicepad_core.audio.microphone.sd.InputStream")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_start_already_recording_raises_error(self, mock_query: Mock, mock_stream_class: Mock) -> None:
        """Test start() raises error if already recording."""
        mock_query.return_value = {"default_samplerate": 16000}
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        stream = MicrophoneStream()
        stream.start()

        with pytest.raises(RuntimeError, match="already recording"):
            stream.start()

    @patch("voicepad_core.audio.microphone.sd.InputStream")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_stop_returns_audio(self, mock_query: Mock, mock_stream_class: Mock) -> None:
        """Test stop() returns accumulated audio."""
        mock_query.return_value = {"default_samplerate": 16000}
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        stream = MicrophoneStream()
        stream.start()

        # Simulate callback adding frames
        stream._frames.append(np.array([[0.1], [0.2]], dtype=np.float32))
        stream._frames.append(np.array([[0.3], [0.4]], dtype=np.float32))

        audio = stream.stop()

        # Verify stream was stopped and closed
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert not stream.is_recording

        # Verify audio is concatenated and flattened
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        np.testing.assert_array_almost_equal(audio, [0.1, 0.2, 0.3, 0.4])

    @patch("voicepad_core.audio.microphone.sd.InputStream")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_stop_not_recording_raises_error(self, mock_query: Mock, mock_stream_class: Mock) -> None:
        """Test stop() raises error if not recording."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()

        with pytest.raises(RuntimeError, match="not recording"):
            stream.stop()

    @patch("voicepad_core.audio.microphone.sd.InputStream")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_stop_empty_frames_returns_zeros(self, mock_query: Mock, mock_stream_class: Mock) -> None:
        """Test stop() returns empty array if no frames recorded."""
        mock_query.return_value = {"default_samplerate": 16000}
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        stream = MicrophoneStream()
        stream.start()
        audio = stream.stop()

        assert audio.dtype == np.float32
        assert len(audio) == 0


class TestMicrophoneStreamSnapshot:
    """Test MicrophoneStream snapshot functionality."""

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_get_snapshot_returns_copy(self, mock_query: Mock) -> None:
        """Test get_snapshot() returns a copy of frames."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        stream._frames.append(np.array([[0.1], [0.2]], dtype=np.float32))
        stream._frames.append(np.array([[0.3], [0.4]], dtype=np.float32))

        snapshot = stream.get_snapshot()

        # Verify snapshot is correct
        assert snapshot.dtype == np.float32
        assert snapshot.ndim == 1
        np.testing.assert_array_almost_equal(snapshot, [0.1, 0.2, 0.3, 0.4])

        # Verify original frames are unchanged
        assert len(stream._frames) == 2

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_get_snapshot_empty_returns_zeros(self, mock_query: Mock) -> None:
        """Test get_snapshot() returns empty array if no frames."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        snapshot = stream.get_snapshot()

        assert snapshot.dtype == np.float32
        assert len(snapshot) == 0

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_get_snapshot_thread_safe(self, mock_query: Mock) -> None:
        """Test get_snapshot() works correctly with internal locking."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        stream._frames.append(np.array([[0.1]], dtype=np.float32))

        # Call get_snapshot - it should work without errors
        # The lock is used internally but we can't easily mock it
        snapshot = stream.get_snapshot()

        # Verify we got the expected snapshot
        assert len(snapshot) == 1
        np.testing.assert_array_equal(snapshot[0], np.array([[0.1]], dtype=np.float32))


class TestMicrophoneStreamCallback:
    """Test MicrophoneStream callback functionality."""

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    @patch("voicepad_core.audio.microphone.sd.CallbackFlags")
    def test_callback_appends_frames(self, mock_flags: Mock, mock_query: Mock) -> None:
        """Test callback appends incoming frames."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        stream._recording = True

        # Simulate callback
        indata = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        stream._callback(indata, 3, None, mock_flags())

        assert len(stream._frames) == 1
        np.testing.assert_array_equal(stream._frames[0], indata)

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    @patch("voicepad_core.audio.microphone.sd.CallbackFlags")
    def test_callback_copies_indata(self, mock_flags: Mock, mock_query: Mock) -> None:
        """Test callback copies indata (not reference)."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        stream._recording = True

        indata = np.array([[0.1], [0.2]], dtype=np.float32)
        stream._callback(indata, 2, None, mock_flags())

        # Modify original indata
        indata[0, 0] = 999.0

        # Stored frame should be unchanged
        assert stream._frames[0][0, 0] == pytest.approx(0.1)

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    @patch("voicepad_core.audio.microphone.sd.CallbackFlags")
    def test_callback_ignores_when_not_recording(self, mock_flags: Mock, mock_query: Mock) -> None:
        """Test callback ignores frames when not recording."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        stream._recording = False

        indata = np.array([[0.1], [0.2]], dtype=np.float32)
        stream._callback(indata, 2, None, mock_flags())

        assert len(stream._frames) == 0


class TestMicrophoneStreamSaveWav:
    """Test MicrophoneStream WAV saving functionality."""

    @patch("voicepad_core.audio.microphone.sf.write")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_save_wav_writes_file(self, mock_query: Mock, mock_write: Mock, tmp_path: Path) -> None:
        """Test save_wav() writes audio to file."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        path = tmp_path / "test.wav"

        stream.save_wav(audio, path)

        # Verify sf.write was called correctly
        mock_write.assert_called_once_with(str(path), audio, 16000, subtype="PCM_16")

    @patch("voicepad_core.audio.microphone.sf.write")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_save_wav_creates_directory(self, mock_query: Mock, mock_write: Mock, tmp_path: Path) -> None:
        """Test save_wav() creates parent directories."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        audio = np.array([0.1, 0.2], dtype=np.float32)
        path = tmp_path / "subdir" / "nested" / "test.wav"

        stream.save_wav(audio, path)

        # Verify directory was created
        assert path.parent.exists()
        mock_write.assert_called_once()

    @patch("voicepad_core.audio.microphone.sf.write")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_save_wav_custom_sample_rate(self, mock_query: Mock, mock_write: Mock, tmp_path: Path) -> None:
        """Test save_wav() with custom sample rate."""
        mock_query.return_value = {"default_samplerate": 16000}

        stream = MicrophoneStream()
        audio = np.array([0.1, 0.2], dtype=np.float32)
        path = tmp_path / "test.wav"

        stream.save_wav(audio, path, sample_rate=48000)

        # Verify custom sample rate was used
        call_args = mock_write.call_args
        assert call_args[0][2] == 48000  # sample_rate argument

    @patch("voicepad_core.audio.microphone.sf.write")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_save_wav_uses_native_rate_by_default(self, mock_query: Mock, mock_write: Mock, tmp_path: Path) -> None:
        """Test save_wav() uses native sample rate by default."""
        mock_query.return_value = {"default_samplerate": 44100}

        stream = MicrophoneStream()
        audio = np.array([0.1, 0.2], dtype=np.float32)
        path = tmp_path / "test.wav"

        stream.save_wav(audio, path)

        # Verify native sample rate was used
        call_args = mock_write.call_args
        assert call_args[0][2] == 44100


class TestMicrophoneStreamProperties:
    """Test MicrophoneStream properties."""

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_sample_rate_property(self, mock_query: Mock) -> None:
        """Test sample_rate property returns native rate."""
        mock_query.return_value = {"default_samplerate": 48000}

        stream = MicrophoneStream()

        assert stream.sample_rate == 48000

    @patch("voicepad_core.audio.microphone.sd.InputStream")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_is_recording_property(self, mock_query: Mock, mock_stream_class: Mock) -> None:
        """Test is_recording property reflects recording state."""
        mock_query.return_value = {"default_samplerate": 16000}
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        stream = MicrophoneStream()

        # Initially not recording
        assert not stream.is_recording

        # After start, should be recording
        stream.start()
        assert stream.is_recording

        # After stop, should not be recording
        stream.stop()
        assert not stream.is_recording
