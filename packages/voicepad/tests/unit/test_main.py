from unittest.mock import patch

from typer.testing import CliRunner
from voicepad.main import app

runner = CliRunner()


def test_toggle_forwards_to_running_tui() -> None:
    with patch("voicepad.tui.control.run_toggle_command", return_value=0) as toggle:
        result = runner.invoke(app, ["toggle"])

    assert result.exit_code == 0
    toggle.assert_called_once_with()


def test_config_show_prints_new_deployment() -> None:
    with patch("voicepad.cli.config.load_config") as load:
        from voicepad.config import AppConfig

        load.return_value = AppConfig()
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "parakeet-v3.transformers-fp16-cuda" in result.stdout
