from __future__ import annotations

from unittest.mock import patch

from voicepad.entrypoint import main


class TestMain:
    def test_toggle_uses_lightweight_control_command(self) -> None:
        """The console entry point dispatches toggle without loading the full CLI."""
        with (
            patch("sys.argv", ["voicepad", "toggle"]),
            patch("voicepad.entrypoint._configure_logging") as configure_logging,
            patch("voicepad.tui.control.run_toggle_command", return_value=0) as run_toggle,
        ):
            result = main()

        assert result == 0
        configure_logging.assert_called_once_with()
        run_toggle.assert_called_once_with()
