from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner
from voicepad.main import app

runner = CliRunner()


class TestToggleRecording:
    def test_toggle_requests_running_tui(self) -> None:
        """The toggle command forwards one request to the running TUI."""
        with patch("voicepad.tui.control.run_toggle_command", return_value=0) as run_toggle:
            result = runner.invoke(app, ["toggle"])

        assert result.exit_code == 0
        run_toggle.assert_called_once_with()

    def test_toggle_reports_missing_tui(self) -> None:
        """The toggle command exits unsuccessfully when no TUI is reachable."""
        with patch("voicepad.tui.control.run_toggle_command", return_value=1):
            result = runner.invoke(app, ["toggle"])

        assert result.exit_code == 1
