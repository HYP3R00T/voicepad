"""Tests for voicepad.main and voicepad.__init__."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner
from voicepad.main import app

runner = CliRunner()


class TestMainApp:
    def test_no_subcommand_launches_tui(self) -> None:
        """When invoked with no subcommand, the TUI run() function is called."""
        with patch("voicepad.tui.app.run") as mock_run:
            result = runner.invoke(app, [])

        mock_run.assert_called_once()
        assert result.exit_code == 0

    def test_help_flag_exits_zero(self) -> None:
        """The --help flag exits with code 0."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_record_subcommand_is_registered(self) -> None:
        """The 'record' subcommand is registered and accessible."""
        result = runner.invoke(app, ["record", "--help"])
        assert result.exit_code == 0

    def test_config_subcommand_is_registered(self) -> None:
        """The 'config' subcommand is registered and accessible."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0


class TestVoicepadInit:
    def test_main_entry_point_calls_app(self) -> None:
        """voicepad.main() delegates to the Typer app."""
        # Patch the app object inside the voicepad package namespace
        with patch("voicepad.app") as mock_app:
            import voicepad

            voicepad.main()

        mock_app.assert_called_once()

    def test_voicepad_main_module_invokes_app(self) -> None:
        """Importing voicepad.__main__ should call the Typer app entry point."""
        with patch("voicepad.main.app") as mock_app:
            import runpy

            runpy.run_module("voicepad.__main__", run_name="__main__")

        mock_app.assert_called_once()
