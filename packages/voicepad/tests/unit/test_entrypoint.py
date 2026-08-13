from __future__ import annotations

import json
import logging
import stat
from unittest.mock import patch

import pytest
from voicepad.entrypoint import _configure_logging, main


def test_configure_logging_uses_private_platform_log_directory(tmp_path) -> None:
    log_dir = tmp_path / "voicepad" / "logs"

    with patch("voicepad.entrypoint.resolve_logs_path", return_value=log_dir):
        log_path = _configure_logging()
        logging.getLogger("voicepad_core.example").info("Core log reached the app session")

    try:
        assert log_path.parent == log_dir
        assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
        assert "Application session started" in log_path.read_text()
        assert "Core log reached the app session" in log_path.read_text()
    finally:
        from utilityhub_logging import cleanup_logging

        cleanup_logging()


class TestMain:
    def test_toggle_uses_lightweight_control_command(self) -> None:
        """The console entry point dispatches toggle without loading the full CLI."""
        with (
            patch("sys.argv", ["voicepad", "toggle"]),
            patch("voicepad.entrypoint._configure_logging") as configure_logging,
            patch("voicepad.entrypoint.cleanup_logging") as cleanup_logging,
            patch("voicepad.tui.control.run_toggle_command", return_value=0) as run_toggle,
        ):
            result = main()

        assert result == 0
        configure_logging.assert_called_once_with()
        run_toggle.assert_called_once_with()
        cleanup_logging.assert_called_once_with()

    def test_session_log_records_completed_outcome(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_dir = tmp_path / "logs"
        with (
            patch("sys.argv", ["voicepad", "toggle"]),
            patch("voicepad.entrypoint.resolve_logs_path", return_value=log_dir),
            patch("voicepad.tui.control.run_toggle_command", return_value=0),
        ):
            main()

        log_path = next(log_dir.glob("*.log"))
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert records[-1]["message"] == "Application session ended: outcome=completed"

    def test_session_log_records_failed_outcome(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_dir = tmp_path / "logs"
        with (
            patch("sys.argv", ["voicepad", "toggle"]),
            patch("voicepad.entrypoint.resolve_logs_path", return_value=log_dir),
            patch("voicepad.tui.control.run_toggle_command", side_effect=RuntimeError("toggle failed")),
            pytest.raises(RuntimeError, match="toggle failed"),
        ):
            main()

        log_path = next(log_dir.glob("*.log"))
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert records[-1]["level"] == "ERROR"
        assert records[-1]["message"] == (
            "Application session ended: outcome=failed error_type=RuntimeError error=toggle failed"
        )

    def test_logging_is_cleaned_up_when_dispatch_fails(self) -> None:
        with (
            patch("sys.argv", ["voicepad", "toggle"]),
            patch("voicepad.entrypoint._configure_logging"),
            patch("voicepad.entrypoint.cleanup_logging") as cleanup_logging,
            patch("voicepad.tui.control.run_toggle_command", side_effect=RuntimeError("failed")),
            pytest.raises(RuntimeError, match="failed"),
        ):
            main()

        cleanup_logging.assert_called_once_with()
