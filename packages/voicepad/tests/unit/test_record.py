from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner
from voicepad.cli.record import _wait_for_stop
from voicepad.main import app
from voicepad_core.audio import WavArtifact


def test_cli_wait_stops_on_capture_failure() -> None:
    microphone = MagicMock()
    microphone.capture_error = RuntimeError("capture failed")
    started = time.monotonic()

    _wait_for_stop(30, microphone)

    assert time.monotonic() - started < 1


def test_record_command_preserves_primary_failure_when_cleanup_fails() -> None:
    microphone = MagicMock()
    microphone.is_recording = True
    microphone.stop.side_effect = RuntimeError("stop failed")
    job = MagicMock()
    job.finish.side_effect = RuntimeError("finish failed")
    runtime = MagicMock()
    runtime.start_recording.return_value = (microphone, job)
    runtime.close.side_effect = RuntimeError("close failed")
    with (
        patch("voicepad.cli.record.ApplicationRuntime", return_value=runtime),
        patch("voicepad.cli.record._wait_for_stop", side_effect=RuntimeError("recording failed")),
    ):
        result = CliRunner().invoke(app, ["record", "start", "--duration", "1"])

    assert result.exit_code == 1
    assert "VoicePad failed: recording failed" in result.stderr
    job.cancel.assert_called_once_with()


def test_no_transcribe_rejects_partial_wav(tmp_path) -> None:  # type: ignore[no-untyped-def]
    microphone = MagicMock()
    microphone.stop.return_value = WavArtifact(tmp_path / "partial.wav", 16_000, 1, 16_000, 1.0)
    microphone.capture_error = RuntimeError("capture failed")
    runtime = MagicMock()
    runtime.start_capture.return_value = microphone
    runtime.stop_capture.return_value = microphone.stop.return_value
    with (
        patch("voicepad.cli.record.ApplicationRuntime", return_value=runtime),
        patch("voicepad.cli.record._wait_for_stop"),
    ):
        result = CliRunner().invoke(app, ["record", "start", "--no-transcribe", "--duration", "1"])

    assert result.exit_code == 2
    assert "Partial WAV preserved" in result.stderr
