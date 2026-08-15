from unittest.mock import MagicMock, patch

from typer.testing import CliRunner
from voicepad.main import app

runner = CliRunner()


def test_toggle_forwards_to_running_tui() -> None:
    with patch("voicepad.tui.control.run_toggle_command", return_value=0) as toggle:
        result = runner.invoke(app, ["toggle"])

    assert result.exit_code == 0
    toggle.assert_called_once_with()


def test_prepare_reports_runtime_cleanup_failure_after_success() -> None:
    runtime = MagicMock()
    runtime.close.side_effect = RuntimeError("close failed")
    with (
        patch("voicepad.main.load_config", return_value=MagicMock()),
        patch("voicepad.main.ApplicationRuntime", return_value=runtime),
    ):
        result = runner.invoke(app, ["prepare"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "close failed"


def test_prepare_preserves_primary_failure_when_runtime_cleanup_fails(caplog) -> None:  # type: ignore[no-untyped-def]
    runtime = MagicMock()
    activation_error = RuntimeError("activation failed")
    runtime.activate.side_effect = activation_error
    runtime.close.side_effect = RuntimeError("close failed")
    with (
        patch("voicepad.main.load_config", return_value=MagicMock()),
        patch("voicepad.main.ApplicationRuntime", return_value=runtime),
    ):
        result = runner.invoke(app, ["prepare"])

    assert result.exit_code == 1
    assert "Preparation failed: activation failed" in result.stderr
    assert activation_error.__notes__ == ["Runtime cleanup also failed: close failed"]
    assert "Runtime cleanup failed after prepare command" in caplog.text


def test_transcribe_preserves_primary_failure_when_runtime_cleanup_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    audio = tmp_path / "audio.wav"
    audio.touch()
    runtime = MagicMock()
    runtime.transcribe_file.side_effect = RuntimeError("transcription failed")
    runtime.close.side_effect = RuntimeError("close failed")
    with (
        patch("voicepad.main.load_config", return_value=MagicMock()),
        patch("voicepad.main.ApplicationRuntime", return_value=runtime),
    ):
        result = runner.invoke(app, ["transcribe", str(audio)])

    assert result.exit_code == 1
    assert "Transcription failed: transcription failed" in result.stderr


def test_config_show_prints_new_deployment() -> None:
    with patch("voicepad.cli.config.load_config") as load:
        from voicepad.config import AppConfig

        load.return_value = AppConfig()
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "parakeet-v3.transformers-fp16-cuda" in result.stdout


def test_config_init_uses_validated_defaults_without_loading_existing_settings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "voicepad.toml"
    with (
        patch("voicepad.cli.config.config_path", return_value=path),
        patch("voicepad.cli.config.load_config") as load,
    ):
        result = runner.invoke(app, ["config", "init"])

    assert result.exit_code == 0
    assert result.stdout.strip() == str(path)
    load.assert_not_called()
    assert 'deployment_id = "parakeet-v3.transformers-fp16-cuda"' in path.read_text()
