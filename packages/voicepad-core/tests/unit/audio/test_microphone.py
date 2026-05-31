"""Tests for microphone.py - MicrophoneSource class."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
from voicepad_core.audio.microphone import MicrophoneSource


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
