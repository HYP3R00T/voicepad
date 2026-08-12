from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

from pytest import MonkeyPatch
from typer.testing import CliRunner
from voicepad.cli.record import _wait_for_stop
from voicepad.config import AppConfig
from voicepad.main import app
from voicepad_core.audio import CaptureFailure, WavArtifact

runner = CliRunner()


class FailedMicrophone:
    capture_failures = (CaptureFailure("capture-write", RuntimeError("writer stopped")),)


def test_timed_wait_stops_immediately_after_capture_failure() -> None:
    started = time.monotonic()

    _wait_for_stop(30.0, FailedMicrophone())

    assert time.monotonic() - started < 1.0


def test_interactive_wait_stops_after_capture_failure(monkeypatch: MonkeyPatch) -> None:
    release_input = threading.Event()

    def blocked_input() -> str:
        release_input.wait(timeout=1)
        return "q"

    monkeypatch.setattr("builtins.input", blocked_input)
    try:
        started = time.monotonic()
        _wait_for_stop(None, FailedMicrophone())
        assert time.monotonic() - started < 1.0
    finally:
        release_input.set()


def test_no_transcribe_reports_partial_wav_as_failure(tmp_path: Path) -> None:
    config = AppConfig(recordings_path=tmp_path)
    artifact = WavArtifact(tmp_path / "partial.wav", 16_000, 1, 16_000, 1.0)
    failure = CaptureFailure("capture-write", RuntimeError("writer stopped"))

    with (
        patch("voicepad.cli.record.load_config", return_value=config),
        patch("voicepad.cli.record.ApplicationRuntime") as runtime_type,
        patch("voicepad.cli.record.MicrophoneStream") as microphone_type,
        patch("voicepad.cli.record._wait_for_stop"),
    ):
        microphone = microphone_type.return_value
        microphone.stop.return_value = artifact
        microphone.capture_failures = (failure,)
        result = runner.invoke(app, ["record", "start", "--no-transcribe", "--duration", "1"])

    assert result.exit_code == 2
    assert "Partial WAV preserved after capture failure" in result.stderr
    runtime_type.return_value.close.assert_called_once_with()
