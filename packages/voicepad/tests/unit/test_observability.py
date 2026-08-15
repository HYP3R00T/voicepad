from __future__ import annotations

import json
import logging
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from utilityhub_logging import LogFormat, cleanup_logging, configure_app_logging
from voicepad.observability import RecordingLogScope


@pytest.fixture(autouse=True)
def cleanup_managed_logging() -> Iterator[None]:
    yield
    cleanup_logging(close_all_loggers=True)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_recording_scope_writes_correlated_private_session_and_recording_logs(tmp_path) -> None:
    session_path = configure_app_logging(
        app_name="voicepad",
        logs_path=tmp_path,
        console=False,
        log_format=LogFormat.JSON,
    )

    with patch("voicepad.observability.resolve_logs_path", return_value=tmp_path):
        scope = RecordingLogScope.start("recording-123")
    scope.logger_for("voicepad_core.pipeline.incremental").info("Pipeline stage completed")
    scope.close(outcome="completed")

    recording_records = _records(scope.path)
    session_records = _records(session_path)
    matching = [record for record in recording_records if record["message"] == "Pipeline stage completed"]

    assert len(matching) == 1
    context = cast("dict[str, object]", matching[0]["context"])
    assert context["recording_id"] == "recording-123"
    assert context["component"] == "voicepad_core.pipeline.incremental"
    assert any(record["message"] == "Pipeline stage completed" for record in session_records)
    assert stat.S_IMODE(scope.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(scope.path.stat().st_mode) == 0o600
    assert not scope._logger.handlers


def test_recording_scope_close_is_idempotent(tmp_path) -> None:
    with patch("voicepad.observability.resolve_logs_path", return_value=tmp_path):
        scope = RecordingLogScope.start("recording-123")

    scope.close(outcome="failed", error=RuntimeError("capture failed"))
    scope.close(outcome="completed")

    assert not scope._logger.handlers
    assert logging.getLogger(scope._logger.name) is scope._logger


def test_concurrent_recording_scopes_do_not_mix_records(tmp_path) -> None:
    session_path = configure_app_logging(
        app_name="voicepad",
        logs_path=tmp_path,
        console=False,
        log_format=LogFormat.JSON,
    )
    with patch("voicepad.observability.resolve_logs_path", return_value=tmp_path):
        first = RecordingLogScope.start("recording-first")
        second = RecordingLogScope.start("recording-second")

    first.logger_for("voicepad_core.audio.microphone").info("FIRST_ONLY")
    second.logger_for("voicepad_core.audio.microphone").info("SECOND_ONLY")
    first.close(outcome="completed")
    second.close(outcome="completed")

    first_text = first.path.read_text()
    second_text = second.path.read_text()
    session_text = session_path.read_text()
    assert "FIRST_ONLY" in first_text and "SECOND_ONLY" not in first_text
    assert "SECOND_ONLY" in second_text and "FIRST_ONLY" not in second_text
    assert "FIRST_ONLY" in session_text and "SECOND_ONLY" in session_text


@pytest.mark.parametrize("outcome", ["completed", "incomplete", "abandoned"])
def test_recording_scope_records_non_failure_outcome(tmp_path, outcome: str) -> None:
    with patch("voicepad.observability.resolve_logs_path", return_value=tmp_path):
        scope = RecordingLogScope.start(f"recording-{outcome}")

    scope.close(outcome=outcome)

    endings = [record for record in _records(scope.path) if str(record["message"]).startswith("Recording scope ended")]
    assert [record["message"] for record in endings] == [f"Recording scope ended: outcome={outcome}"]
    assert endings[0]["level"] == "INFO"


def test_recording_scope_records_failure_with_exception(tmp_path) -> None:
    with patch("voicepad.observability.resolve_logs_path", return_value=tmp_path):
        scope = RecordingLogScope.start("recording-failed")

    try:
        raise RuntimeError("capture failed")
    except RuntimeError as error:
        scope.close(outcome="failed", error=error)

    ending = _records(scope.path)[-1]
    assert ending["level"] == "ERROR"
    assert ending["message"] == ("Recording scope ended: outcome=failed error_type=RuntimeError error=capture failed")
    assert "RuntimeError: capture failed" in str(ending["exception"])


def test_scope_start_closes_handler_when_private_mode_setup_fails(tmp_path) -> None:
    def reject_log_file(path: Path, mode: int) -> None:
        del mode
        if Path(path).suffix == ".log":
            raise PermissionError("chmod failed")

    with (
        patch("voicepad.observability.resolve_logs_path", return_value=tmp_path),
        patch("voicepad.observability.os.chmod", side_effect=reject_log_file),
        pytest.raises(PermissionError, match="chmod failed"),
    ):
        RecordingLogScope.start("recording-permission-failure")

    failed_logger = logging.getLogger("voicepad.recording.recording-permission-failure")
    assert not failed_logger.handlers
