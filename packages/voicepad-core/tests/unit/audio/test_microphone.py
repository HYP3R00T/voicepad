"""Tests for disk-backed microphone capture."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import numpy as np
import pytest
import sounddevice as sd
from voicepad_core.audio import (
    AudioStreamStateError,
    AudioWindow,
    AudioWriteBackpressureError,
    MicrophoneStream,
    WavArtifact,
)


@pytest.fixture(autouse=True)
def input_backend() -> Iterator[tuple[Mock, Mock, Mock]]:
    """Keep device discovery and format checks independent of host audio hardware."""
    with (
        patch(
            "voicepad_core.audio.microphone.sd.query_devices",
            return_value={"index": 25, "name": "default", "hostapi": 0, "default_samplerate": 48_000},
        ) as devices,
        patch("voicepad_core.audio.microphone.sd.query_hostapis", return_value={"name": "ALSA"}) as host_apis,
        patch("voicepad_core.audio.microphone.sd.check_input_settings") as check,
        patch("voicepad_core.audio.microphone.sys.platform", "linux"),
    ):
        yield devices, host_apis, check


def test_construction_does_not_query_hardware(input_backend: tuple[Mock, Mock, Mock], tmp_path: Path) -> None:
    stream = MicrophoneStream(tmp_path / "recording.wav", device_index=2)

    for query in input_backend:
        query.assert_not_called()
    assert stream.sample_rate == 16_000
    assert stream.signal_health.samples == 0


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_start_checks_endpoint_then_opens_writer_before_microphone(
    input_stream_type: Mock,
    recording_type: Mock,
    input_backend: tuple[Mock, Mock, Mock],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    devices, host_apis, check = input_backend
    calls = Mock()
    calls.attach_mock(check, "check")
    calls.attach_mock(recording_type.return_value.start, "writer_start")
    calls.attach_mock(input_stream_type, "open_stream")
    stream = MicrophoneStream(tmp_path / "recording.wav", device_index=3)

    with caplog.at_level(logging.INFO):
        stream.start()

    devices.assert_called_once_with(None, "input")
    host_apis.assert_called_once_with(0)
    check.assert_called_once_with(device=25, channels=1, dtype="float32", samplerate=16_000)
    recording_type.assert_called_once_with(tmp_path / "recording.wav", 16_000, 1, logger=ANY, log_context={})
    recording_type.return_value.start.assert_called_once_with()
    input_stream_type.assert_called_once_with(
        samplerate=16_000,
        channels=1,
        dtype="float32",
        device=25,
        callback=stream._callback,
        finished_callback=stream._stream_finished,
    )
    input_stream_type.return_value.start.assert_called_once_with()
    assert [call[0] for call in calls.mock_calls][:3] == ["check", "writer_start", "open_stream"]
    assert "requested=system-default device_index=25 device_name=default host_api=ALSA" in caplog.text
    assert "default_sample_rate=48000 requested_sample_rate=16000 channels=1 dtype=float32" in caplog.text
    assert stream.is_recording


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_non_linux_explicit_input_is_resolved_and_opened(
    input_stream_type: Mock,
    _recording_type: Mock,
    input_backend: tuple[Mock, Mock, Mock],
    tmp_path: Path,
) -> None:
    devices, _, check = input_backend
    devices.return_value["index"] = 3
    with patch("voicepad_core.audio.microphone.sys.platform", "darwin"):
        stream = MicrophoneStream(tmp_path / "recording.wav", device_index=3)
        stream.start()

    devices.assert_called_once_with(3, "input")
    check.assert_called_once_with(device=3, channels=1, dtype="float32", samplerate=16_000)
    assert input_stream_type.call_args.kwargs["device"] == 3


@pytest.mark.parametrize("stage", ["discovery", "format"])
@pytest.mark.parametrize("error_type", [sd.PortAudioError, ValueError])
@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_preflight_failure_does_not_create_writer_or_open_stream(
    input_stream_type: Mock,
    recording_type: Mock,
    input_backend: tuple[Mock, Mock, Mock],
    tmp_path: Path,
    stage: str,
    error_type: type[Exception],
) -> None:
    devices, _, check = input_backend
    error = error_type("input unavailable")
    (devices if stage == "discovery" else check).side_effect = error
    destination = tmp_path / "recordings" / "recording.wav"
    stream = MicrophoneStream(destination)

    with pytest.raises(AudioStreamStateError, match="system sound settings") as exc:
        stream.start()

    assert exc.value.__cause__ is error
    if stage == "format":
        assert "'default' (ALSA, index 25) cannot use 16000 Hz, 1 channel(s), float32" in str(exc.value)
    recording_type.assert_not_called()
    input_stream_type.assert_not_called()
    assert not destination.parent.exists()
    assert not stream.is_recording


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_start_rechecks_default_after_failed_preflight(
    input_stream_type: Mock,
    _recording_type: Mock,
    input_backend: tuple[Mock, Mock, Mock],
    tmp_path: Path,
) -> None:
    devices, _, check = input_backend
    check.side_effect = [sd.PortAudioError("unavailable"), None]
    stream = MicrophoneStream(tmp_path / "recording.wav")
    with pytest.raises(AudioStreamStateError):
        stream.start()
    devices.return_value["index"] = 26

    stream.start()

    assert devices.call_count == 2
    assert input_stream_type.call_args.kwargs["device"] == 26


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream", side_effect=RuntimeError("open failed"))
def test_start_failure_aborts_writer(
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
def test_unavailable_system_microphone_has_linux_guidance(
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
def test_stop_finalizes_recording(
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


def test_stop_before_start_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AudioStreamStateError, match="not recording"):
        MicrophoneStream(tmp_path / "recording.wav").stop()


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_read_window_uses_absolute_sample_position(
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
def test_callback_copies_audio_to_writer(
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    stream = MicrophoneStream(tmp_path / "recording.wav")
    stream.start()
    samples = np.array([[0.1], [0.2]], dtype=np.float32)

    stream._callback(samples, 2, None, sd.CallbackFlags())
    samples[0, 0] = 1.0

    written = recording_type.return_value.append.call_args.args[0]
    np.testing.assert_allclose(written[:, 0], [0.1, 0.2])


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_callback_failure_aborts_capture(
    input_stream_type: Mock,
    recording_type: Mock,
    tmp_path: Path,
) -> None:
    recording_type.return_value.append.side_effect = RuntimeError("writer failed")
    stream = MicrophoneStream(tmp_path / "recording.wav")
    stream.start()

    with pytest.raises(sd.CallbackAbort):
        stream._callback(np.zeros((1, 1), dtype=np.float32), 1, None, sd.CallbackFlags())

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
def test_disk_backed_capture_reads_and_finalizes(input_stream_type: Mock, tmp_path: Path) -> None:
    destination = tmp_path / "recording.wav"
    stream = MicrophoneStream(destination)
    stream.start()
    stream._callback(
        np.array([[0.0], [0.25], [0.5]], dtype=np.float32),
        3,
        None,
        sd.CallbackFlags(),
    )

    window = stream.read_window(1)
    artifact = stream.stop()

    assert (window.start_sample, window.end_sample) == (1, 3)
    np.testing.assert_allclose(window.samples, [0.25, 0.5])
    assert (artifact.path, artifact.frame_count, destination.exists()) == (destination, 3, True)
    assert stream.signal_health.peak == 0.5
    assert stream.signal_health.samples == 3
    assert stream.signal_health.warnings == ()
    assert stream.discontinuity_warnings == ()


@pytest.mark.parametrize("flag", ["input_overflow", "input_underflow"])
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_input_discontinuities_are_counted_without_stopping_or_discarding_delivered_audio(
    _input_stream_type: Mock, tmp_path: Path, caplog: pytest.LogCaptureFixture, flag: str
) -> None:
    import soundfile as sf

    stream = MicrophoneStream(tmp_path / "gaps.wav")
    stream.start()
    status = sd.CallbackFlags()
    setattr(status, flag, True)
    block = np.array([[0.25], [-0.5]], dtype=np.float32)
    stream._callback(block, 2, None, status)
    stream._callback(block, 2, None, sd.CallbackFlags())
    stream._callback(block, 2, None, status)

    assert stream.is_recording
    assert stream.capture_error is None
    assert stream.input_overflow_count == (2 if flag == "input_overflow" else 0)
    assert stream.input_underflow_count == (2 if flag == "input_underflow" else 0)
    assert "2 callback event(s)" in stream.discontinuity_warnings[0]
    assert "duration unknown" in stream.discontinuity_warnings[0]
    assert "Microphone callback status" not in caplog.text

    with caplog.at_level(logging.INFO):
        artifact = stream.stop()
    persisted, _ = sf.read(artifact.path, dtype="float32")
    np.testing.assert_array_equal(persisted, np.tile(block[:, 0], 3))
    assert artifact.frame_count == 6  # No guessed padding for unknown lost samples.
    assert "Microphone callback status summary" in caplog.text
    assert "Microphone discontinuity" in caplog.text
    assert "missing_duration_s" not in caplog.text
    stream._callback(block, 2, None, status)  # Late callback after stop is ignored.
    assert "2 callback event(s)" in stream.discontinuity_warnings[0]


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_discontinuity_counts_reset_on_a_new_recording(
    _input_stream_type: Mock, recording_type: Mock, tmp_path: Path
) -> None:
    recording_type.return_value.finish.return_value = WavArtifact(tmp_path / "recording.wav", 16_000, 1, 1, 1 / 16_000)
    stream = MicrophoneStream(tmp_path / "recording.wav")
    status = sd.CallbackFlags()
    status.input_overflow = status.input_underflow = True
    stream.start()
    stream._callback(np.zeros((1, 1), dtype=np.float32), 1, None, status)
    assert len(stream.discontinuity_warnings) == 2
    assert stream.input_overflow_count == stream.input_underflow_count == 1
    stream.stop()
    stream.start()
    assert stream.input_overflow_count == stream.input_underflow_count == 0
    assert stream.discontinuity_warnings == ()


@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_unexpected_stream_end_preserves_delivered_audio(_input_stream_type: Mock, tmp_path: Path) -> None:
    stream = MicrophoneStream(tmp_path / "disconnected.wav")
    stream.start()
    stream._callback(np.full((4, 1), 0.25, dtype=np.float32), 4, None, sd.CallbackFlags())
    stream._stream_finished()  # Simulate device loss reported by PortAudio.
    artifact = stream.stop()

    assert "stopped unexpectedly" in str(stream.capture_error)
    assert artifact.path.exists()
    assert artifact.frame_count == 4
    np.testing.assert_array_equal(stream.read_window(0).samples, np.full(4, 0.25))


@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_non_input_status_is_logged_but_not_labeled_input_loss(
    _input_stream_type: Mock, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stream = MicrophoneStream(tmp_path / "status.wav")
    stream.start()
    status = sd.CallbackFlags()
    status.output_underflow = True
    stream._callback(np.zeros((1, 1), dtype=np.float32), 1, None, status)
    stream.stop()
    stream._stream_finished()  # Expected completion must not become a capture error.

    assert stream.capture_error is None
    assert stream.discontinuity_warnings == ()
    assert "output underflow" in caplog.text


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_callback_failure_is_logged_only_when_capture_stops(
    _input_stream_type: Mock, recording_type: Mock, tmp_path: Path
) -> None:
    logger = Mock()
    error = AudioWriteBackpressureError("writer queue full")
    recording_type.return_value.append.side_effect = error
    artifact = WavArtifact(tmp_path / "recording.wav", 16_000, 1, 4, 4 / 16_000)
    recording_type.return_value.finish.return_value = artifact
    stream = MicrophoneStream(artifact.path, logger=logger)
    stream.start()
    logger.reset_mock()

    with pytest.raises(sd.CallbackAbort):
        stream._callback(np.zeros((1, 1), dtype=np.float32), 1, None, sd.CallbackFlags())
    stream._stream_finished()

    assert stream.capture_error is error
    logger.assert_not_called()
    assert logger.method_calls == []  # No logging handler can block the native callback.
    assert stream.stop() == artifact
    logger.error.assert_called_once()
    assert logger.error.call_args.args[-1] is error


@patch("voicepad_core.audio.microphone.LiveWavRecording")
@patch("voicepad_core.audio.microphone.sd.InputStream")
def test_finished_callback_defers_error_logging(_input_stream_type: Mock, recording_type: Mock, tmp_path: Path) -> None:
    logger = Mock()
    artifact = WavArtifact(tmp_path / "recording.wav", 16_000, 1, 0, 0.0)
    recording_type.return_value.finish.return_value = artifact
    stream = MicrophoneStream(artifact.path, logger=logger)
    stream.start()
    logger.reset_mock()
    stream._stream_finished()

    assert "stopped unexpectedly" in str(stream.capture_error)
    assert logger.method_calls == []
    stream.stop()
    logger.error.assert_called_once()
