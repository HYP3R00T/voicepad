"""Tests for microphone.py - MicrophoneStream and MicrophoneSource classes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from voicepad_core.audio.microphone import MicrophoneSource, MicrophoneStream


class TestMicrophoneInitialization:
    """Test MicrophoneSource initialization and device query."""

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_init_default_device(self, mock_query: Mock) -> None:
        """Test initialization with default device (device_index=None)."""
        mock_query.return_value = {"default_samplerate": 48000}

        source = MicrophoneSource()

        mock_query.assert_called_once_with(None, kind="input")
        assert source.get_sample_rate() == 48000
        assert source.get_channels() == 1
        assert source._device_index is None
        assert source._duration_s == 5.0

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_init_specific_device(self, mock_query: Mock) -> None:
        """Test initialization with specific device index."""
        mock_query.return_value = {"default_samplerate": 44100}

        source = MicrophoneSource(device_index=1)

        mock_query.assert_called_once_with(1, kind="input")
        assert source.get_sample_rate() == 44100
        assert source._device_index == 1

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_init_custom_duration(self, mock_query: Mock) -> None:
        """Test initialization with custom duration."""
        mock_query.return_value = {"default_samplerate": 16000}

        source = MicrophoneSource(duration_s=10.0)

        assert source._duration_s == 10.0


class TestDeviceQuery:
    """Test device query and sample rate detection."""

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_query_native_rate_success(self, mock_query: Mock) -> None:
        """Test successful device query returns correct sample rate."""
        mock_query.return_value = {"default_samplerate": 48000}

        source = MicrophoneSource()

        assert source.get_sample_rate() == 48000

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_query_failure_fallback(self, mock_query: Mock) -> None:
        """Test query failure falls back to 16000 Hz."""
        mock_query.side_effect = Exception("Device not found")

        source = MicrophoneSource()

        assert source.get_sample_rate() == 16000

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_invalid_device_index_fallback(self, mock_query: Mock) -> None:
        """Test invalid device index falls back to 16000 Hz."""
        mock_query.side_effect = ValueError("Invalid device index")

        source = MicrophoneSource(device_index=999)

        assert source.get_sample_rate() == 16000

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_zero_sample_rate_fallback(self, mock_query: Mock) -> None:
        """Test zero sample rate falls back to 16000 Hz."""
        mock_query.return_value = {"default_samplerate": 0}

        source = MicrophoneSource()

        assert source.get_sample_rate() == 16000

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_negative_sample_rate_fallback(self, mock_query: Mock) -> None:
        """Test negative sample rate falls back to 16000 Hz."""
        mock_query.return_value = {"default_samplerate": -1}

        source = MicrophoneSource()

        assert source.get_sample_rate() == 16000

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_missing_samplerate_key_fallback(self, mock_query: Mock) -> None:
        """Test missing default_samplerate key falls back to 16000 Hz."""
        mock_query.return_value = {}

        source = MicrophoneSource()

        assert source.get_sample_rate() == 16000


class TestRecording:
    """Test audio recording functionality."""

    @patch("voicepad_core.audio.microphone.sd.wait")
    @patch("voicepad_core.audio.microphone.sd.rec")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_read_records_audio(self, mock_query: Mock, mock_rec: Mock, mock_wait: Mock) -> None:
        """Test successful recording returns float32 array."""
        mock_query.return_value = {"default_samplerate": 16000}
        # sd.rec returns (N, 1) shaped array
        mock_rec.return_value = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)

        source = MicrophoneSource(duration_s=1.0)
        audio = source.read()

        # Verify sd.rec was called with correct parameters
        mock_rec.assert_called_once_with(
            frames=16000,  # 1.0 * 16000
            samplerate=16000,
            channels=1,
            dtype="float32",
            device=None,
        )
        # Verify sd.wait was called to block until complete
        mock_wait.assert_called_once()
        # Verify output is float32
        assert audio.dtype == np.float32
        # Verify output is flattened to 1D
        assert audio.ndim == 1
        assert len(audio) == 5

    @patch("voicepad_core.audio.microphone.sd.wait")
    @patch("voicepad_core.audio.microphone.sd.rec")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_recording_duration_matches_requested(self, mock_query: Mock, mock_rec: Mock, mock_wait: Mock) -> None:
        """Test recording duration matches requested duration."""
        mock_query.return_value = {"default_samplerate": 48000}
        expected_frames = int(5.0 * 48000)  # 240000 frames
        mock_rec.return_value = np.zeros((expected_frames, 1), dtype=np.float32)

        source = MicrophoneSource(duration_s=5.0)
        audio = source.read()

        # Verify correct frame count
        mock_rec.assert_called_once()
        call_args = mock_rec.call_args
        assert call_args.kwargs["frames"] == expected_frames
        assert len(audio) == expected_frames

    @patch("voicepad_core.audio.microphone.sd.wait")
    @patch("voicepad_core.audio.microphone.sd.rec")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_output_flattened_to_1d(self, mock_query: Mock, mock_rec: Mock, mock_wait: Mock) -> None:
        """Test output is flattened from (N, 1) to (N,)."""
        mock_query.return_value = {"default_samplerate": 16000}
        # sd.rec returns (N, 1) shape for mono
        mock_rec.return_value = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)

        source = MicrophoneSource()
        audio = source.read()

        # Verify shape is (N,) not (N, 1)
        assert audio.shape == (4,)
        assert audio.ndim == 1
        # Verify values are preserved
        np.testing.assert_array_almost_equal(audio, [0.1, 0.2, 0.3, 0.4])

    @patch("voicepad_core.audio.microphone.sd.wait")
    @patch("voicepad_core.audio.microphone.sd.rec")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_sd_wait_called(self, mock_query: Mock, mock_rec: Mock, mock_wait: Mock) -> None:
        """Test sd.wait() is called to block until recording complete."""
        mock_query.return_value = {"default_samplerate": 16000}
        mock_rec.return_value = np.array([[0.1]], dtype=np.float32)

        source = MicrophoneSource()
        source.read()

        # Verify sd.wait was called exactly once
        mock_wait.assert_called_once()

    @patch("voicepad_core.audio.microphone.sd.wait")
    @patch("voicepad_core.audio.microphone.sd.rec")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_recording_with_specific_device(self, mock_query: Mock, mock_rec: Mock, mock_wait: Mock) -> None:
        """Test recording with specific device index."""
        mock_query.return_value = {"default_samplerate": 44100}
        mock_rec.return_value = np.array([[0.5]], dtype=np.float32)

        source = MicrophoneSource(device_index=2)
        source.read()

        # Verify device parameter passed to sd.rec
        call_args = mock_rec.call_args
        assert call_args.kwargs["device"] == 2

    @patch("voicepad_core.audio.microphone.sd.wait")
    @patch("voicepad_core.audio.microphone.sd.rec")
    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_custom_duration_frame_calculation(self, mock_query: Mock, mock_rec: Mock, mock_wait: Mock) -> None:
        """Test frame calculation with custom duration."""
        mock_query.return_value = {"default_samplerate": 48000}
        mock_rec.return_value = np.zeros((480000, 1), dtype=np.float32)

        source = MicrophoneSource(duration_s=10.0)
        source.read()

        # Verify frames = duration_s * sample_rate
        # 10.0 * 48000 = 480000
        call_args = mock_rec.call_args
        assert call_args.kwargs["frames"] == 480000


class TestProperties:
    """Test property methods."""

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_get_sample_rate_returns_queried_rate(self, mock_query: Mock) -> None:
        """Test get_sample_rate() returns queried native rate."""
        mock_query.return_value = {"default_samplerate": 44100}

        source = MicrophoneSource()

        assert source.get_sample_rate() == 44100

    @patch("voicepad_core.audio.microphone.sd.query_devices")
    def test_get_channels_always_returns_mono(self, mock_query: Mock) -> None:
        """Test get_channels() always returns 1 (mono)."""
        mock_query.return_value = {"default_samplerate": 48000}

        source = MicrophoneSource()

        assert source.get_channels() == 1


# ============================================================================
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
