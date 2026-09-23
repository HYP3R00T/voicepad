from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner
from voicepad.cli.record import _wait_for_stop
from voicepad.config import AppConfig
from voicepad.main import app
from voicepad_core.audio import SignalHealth, WavArtifact


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
        patch("voicepad.cli.record.load_config", return_value=AppConfig()),
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
        patch("voicepad.cli.record.load_config", return_value=AppConfig()),
        patch("voicepad.cli.record.ApplicationRuntime", return_value=runtime),
        patch("voicepad.cli.record._wait_for_stop"),
    ):
        result = CliRunner().invoke(app, ["record", "start", "--no-transcribe", "--duration", "1"])

    assert result.exit_code == 2
    assert "Partial WAV preserved" in result.stderr


def test_no_transcribe_prints_signal_warning_without_failing(tmp_path: Path) -> None:
    microphone = MagicMock(capture_error=None, discontinuity_warnings=())
    microphone.signal_health = SignalHealth().with_samples(np.zeros(32, dtype=np.float32), 16)
    runtime = MagicMock()
    runtime.start_capture.return_value = microphone
    runtime.stop_capture.return_value = WavArtifact(tmp_path / "quiet.wav", 16, 1, 32, 2.0)
    with (
        patch("voicepad.cli.record.load_config", return_value=AppConfig()),
        patch("voicepad.cli.record.ApplicationRuntime", return_value=runtime),
        patch("voicepad.cli.record._wait_for_stop"),
    ):
        result = CliRunner().invoke(app, ["record", "start", "--no-transcribe", "--duration", "2"])

    assert result.exit_code == 0
    assert "near-silent" in result.stderr
    assert "Saved WAV:" in result.stdout


@pytest.mark.parametrize("no_transcribe", [True, False])
def test_discontinuous_recording_is_saved_but_not_reported_complete_or_copied(
    tmp_path: Path, no_transcribe: bool
) -> None:
    warning = "audio input overflow: 1 callback event(s); audio was discarded (lost duration unknown)."
    microphone = MagicMock(capture_error=None, discontinuity_warnings=(warning,))
    microphone.signal_health = SignalHealth()
    artifact = WavArtifact(tmp_path / "gaps.wav", 16_000, 1, 16_000, 1.0)
    transcription = MagicMock(complete=False, text="partial text", warnings=(warning,))
    runtime = MagicMock()
    runtime.start_capture.return_value = microphone
    runtime.start_recording.return_value = (microphone, MagicMock())
    runtime.stop_capture.return_value = artifact
    runtime.stop_recording.return_value = (artifact, transcription)
    with (
        patch("voicepad.cli.record.load_config", return_value=AppConfig()),
        patch("voicepad.cli.record.ApplicationRuntime", return_value=runtime),
        patch("voicepad.cli.record._wait_for_stop"),
        patch("voicepad.cli.record.persist_markdown", return_value=tmp_path / "gaps.md") as persist,
        patch("voicepad.cli.record.copy_to_clipboard") as copy,
    ):
        args = ["record", "start", "--duration", "1"]
        if no_transcribe:
            args.append("--no-transcribe")
        result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert warning in result.stderr
    assert "Saved WAV:" not in result.stdout
    copy.assert_not_called()
    runtime.end_recording.assert_called_once_with(outcome="incomplete")
    if no_transcribe:
        assert "Partial WAV preserved" in result.stderr
        runtime.stop_capture.assert_called_once_with(microphone)
        persist.assert_not_called()
    else:
        persist.assert_called_once_with(artifact.path, transcription, AppConfig().markdown_path)
        assert "partial text" in result.stdout
        assert "incomplete" in result.stderr
