"""Tests for disk-backed microphone capture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, patch

import numpy as np
import pytest
import sounddevice as sd
from voicepad_core.audio import AudioStreamStateError, AudioWindow, MicrophoneStream, WavArtifact


@patch("voicepad_core.audio.microphone.sd.query_devices")
def test_capture_uses_canonical_sample_rate_without_device_query(mock_query: Mock, tmp_path: Path) -> None:
    stream = MicrophoneStream(tmp_path / "recording.wav", device_index=2)

    mock_query.assert_not_called()
    assert stream.sample_rate == 16_000


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
@patch("voicepad_core.audio.microphone.sd.query_devices")
def test_start_opens_writer_before_microphone(
    mock_query: Mock,
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    mock_query.return_value = {"default_samplerate": 48_000}
    native_stream = input_stream_type.return_value
    stream = MicrophoneStream(tmp_path / "recording.wav", device_index=3)

    stream.start()

    recording_type.assert_called_once_with(
        tmp_path / "recording.wav",
        16_000,
        1,
        logger=ANY,
        log_context={},
    )
    recording_type.return_value.start.assert_called_once_with()
    input_stream_type.assert_called_once_with(
        samplerate=16_000,
        channels=1,
        dtype="float32",
        device=None,
        callback=stream._callback,
        finished_callback=stream._stream_finished,
    )
    native_stream.start.assert_called_once_with()
    assert stream.is_recording


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream", side_effect=RuntimeError("open failed"))
@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_start_failure_aborts_writer(
    mock_query: Mock,
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    stream = MicrophoneStream(tmp_path / "recording.wav")

    with pytest.raises(RuntimeError, match="open failed"):
        stream.start()

    recording_type.return_value.abort.assert_called_once_with()
    assert not stream.is_recording


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch(
    "voicepad_core.audio.microphone.sd.InputStream",
    side_effect=RuntimeError("Error opening InputStream: Device unavailable [PaErrorCode -9985]"),
)
@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_unavailable_system_microphone_has_linux_guidance(
    mock_query: Mock,
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    stream = MicrophoneStream(tmp_path / "recording.wav")

    with pytest.raises(AudioStreamStateError, match="Linux sound settings"):
        stream.start()

    recording_type.return_value.abort.assert_called_once_with()


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_stop_finalizes_recording(
    mock_query: Mock,
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    artifact = WavArtifact(tmp_path / "recording.wav", 16_000, 1, 32_000, 2.0)
    recording_type.return_value.finish.return_value = artifact
    stream = MicrophoneStream(artifact.path)
    stream.start()

    assert stream.stop() == artifact

    input_stream_type.return_value.stop.assert_called_once_with()
    input_stream_type.return_value.close.assert_called_once_with()
    recording_type.return_value.finish.assert_called_once_with()
    assert not stream.is_recording


@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_stop_before_start_is_rejected(mock_query: Mock, tmp_path: Path) -> None:
    with pytest.raises(AudioStreamStateError, match="not recording"):
        MicrophoneStream(tmp_path / "recording.wav").stop()


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_read_window_uses_absolute_sample_position(
    mock_query: Mock,
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    expected = AudioWindow(np.array([0.25, 0.5], dtype=np.float32), 12)
    recording_type.return_value.read_from.return_value = expected
    stream = MicrophoneStream(tmp_path / "recording.wav")
    stream.start()

    assert stream.read_window(12, 20) is expected
    recording_type.return_value.read_from.assert_called_once_with(12, 20)


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_callback_copies_audio_to_writer(
    mock_query: Mock,
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    stream = MicrophoneStream(tmp_path / "recording.wav")
    stream.start()
    samples = np.array([[0.1], [0.2]], dtype=np.float32)

    stream._callback(samples, 2, None, MagicMock(__bool__=Mock(return_value=False)))
    samples[0, 0] = 1.0

    written = recording_type.return_value.append.call_args.args[0]
    np.testing.assert_allclose(written[:, 0], [0.1, 0.2])


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_callback_failure_aborts_capture(
    mock_query: Mock,
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    recording_type.return_value.append.side_effect = RuntimeError("writer failed")
    stream = MicrophoneStream(tmp_path / "recording.wav")
    stream.start()

    with pytest.raises(sd.CallbackAbort):
        stream._callback(np.zeros((1, 1), dtype=np.float32), 1, None, MagicMock())

    assert str(stream.capture_error) == "writer failed"


@pytest.mark.parametrize("method", ("stop", "close"))
@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_native_cleanup_failure_still_finalizes_audio(
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
    method: str,
) -> None:
    artifact = WavArtifact(tmp_path / "recording.wav", 16_000, 1, 16_000, 1.0)
    recording_type.return_value.finish.return_value = artifact
    getattr(input_stream_type.return_value, method).side_effect = RuntimeError(f"{method} failed")
    stream = MicrophoneStream(artifact.path)
    stream.start()

    assert stream.stop() == artifact
    assert str(stream.capture_error) == f"{method} failed"
    recording_type.return_value.finish.assert_called_once_with()


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_unexpected_stream_end_is_observable(
    _input_stream_type: Mock,
    _recording_type: Mock,
    tmp_path: Path,
) -> None:
    stream = MicrophoneStream(tmp_path / "recording.wav")
    stream.start()
    stream._stream_finished()

    assert "stopped unexpectedly" in str(stream.capture_error)


@patch("voicepad_core.audio.microphone.sd.InputStream")
@patch("voicepad_core.audio.microphone.sd.query_devices", return_value={"default_samplerate": 16_000})
def test_disk_backed_capture_reads_and_finalizes(
    mock_query: Mock,
    input_stream_type: Mock,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "recording.wav"
    stream = MicrophoneStream(destination)
    stream.start()
    stream._callback(
        np.array([[0.0], [0.25], [0.5]], dtype=np.float32),
        3,
        None,
        MagicMock(__bool__=Mock(return_value=False)),
    )

    window = stream.read_window(1)
    artifact = stream.stop()

    assert (window.start_sample, window.end_sample) == (1, 3)
    np.testing.assert_allclose(window.samples, [0.25, 0.5])
    assert (artifact.path, artifact.frame_count, destination.exists()) == (destination, 3, True)
