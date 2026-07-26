from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from voicepad.tui.handlers.settings_handler import SettingsHandler
from voicepad_core.config import Config


class TestOnSettingsSave:
    """Tests for on_settings_save method."""

    def _build_app(self, config: Config, selector_map: dict[str, object]) -> Mock:
        app = Mock()
        app.config = config
        app._hotkey_listener = None
        app._overlay = None
        app._recording = False
        app._transcribing = False
        app._model_ready = True
        app._start_hotkey_listener = Mock()
        app._warm_model_worker = Mock()
        app._set_status = Mock()
        app.theme = "tokyo-night"
        app.set_timer = Mock()

        def _query_one(selector: str, _widget_type: object) -> object:
            return selector_map[selector]

        app.query_one.side_effect = _query_one
        return app

    def _build_selector_map(
        self,
        recordings_path: Path,
        markdown_path: Path,
        status: Mock,
        vad_model_path: Path | None = None,
    ) -> dict[str, object]:
        recordings_input = Mock()
        recordings_input.value = str(recordings_path)
        markdown_input = Mock()
        markdown_input.value = str(markdown_path)
        vad_input = Mock()
        vad_input.value = str(vad_model_path) if vad_model_path else ""

        model_select = Mock()
        model_select.value = "turbo"

        device_select = Mock()
        device_select.value = -1

        theme_select = Mock()
        theme_select.value = "tokyo-night"

        ctrl_checkbox = Mock()
        ctrl_checkbox.value = False
        alt_checkbox = Mock()
        alt_checkbox.value = False
        shift_checkbox = Mock()
        shift_checkbox.value = False
        cmd_checkbox = Mock()
        cmd_checkbox.value = False

        key_select = Mock()
        key_select.value = "v"

        return {
            "#settings-status": status,
            "#setting-recordings_path": recordings_input,
            "#setting-markdown_path": markdown_input,
            "#setting-vad_model_path": vad_input,
            "#setting-transcription_model": model_select,
            "#setting-input_device_index": device_select,
            "#setting-theme": theme_select,
            "#hotkey-mod-ctrl": ctrl_checkbox,
            "#hotkey-mod-alt": alt_checkbox,
            "#hotkey-mod-shift": shift_checkbox,
            "#hotkey-mod-cmd": cmd_checkbox,
            "#hotkey-key-select": key_select,
        }

    def test_creates_missing_recordings_and_markdown_dirs(self, tmp_path: Path) -> None:
        """Saving settings creates the configured directories when they do not exist."""
        recordings_path = tmp_path / "custom" / "recordings"
        markdown_path = tmp_path / "notes" / "markdown"
        vad_model_path = tmp_path / "models" / "vad"
        config = Config(
            recordings_path=recordings_path,
            markdown_path=markdown_path,
            vad_model_path=vad_model_path,
        )
        status = Mock()
        selector_map = self._build_selector_map(recordings_path, markdown_path, status, vad_model_path)
        app = self._build_app(config, selector_map)
        handler = SettingsHandler(app)

        with (
            patch("utilityhub_config.get_config_path", return_value=tmp_path / "voicepad.yaml"),
            patch("utilityhub_config.write_config") as mock_write_config,
        ):
            handler.on_settings_save()

        assert recordings_path.exists()
        assert markdown_path.exists()
        assert vad_model_path.exists()
        mock_write_config.assert_called_once()
        status.update.assert_called()

    def test_reports_error_when_recordings_path_cannot_be_created(self, tmp_path: Path) -> None:
        """Saving settings shows an error if a directory path cannot be created."""
        recordings_path = tmp_path / "blocked"
        recordings_path.write_text("not a directory", encoding="utf-8")
        markdown_path = tmp_path / "notes" / "markdown"
        vad_model_path = tmp_path / "models" / "vad"
        config = Config(
            recordings_path=recordings_path,
            markdown_path=markdown_path,
            vad_model_path=vad_model_path,
        )
        status = Mock()
        selector_map = self._build_selector_map(recordings_path, markdown_path, status, vad_model_path)
        app = self._build_app(config, selector_map)
        handler = SettingsHandler(app)

        with (
            patch("utilityhub_config.get_config_path", return_value=tmp_path / "voicepad.yaml"),
            patch("utilityhub_config.write_config") as mock_write_config,
        ):
            handler.on_settings_save()

        mock_write_config.assert_not_called()
        status.update.assert_called()

    def test_vad_model_path_is_saved_correctly(self, tmp_path: Path) -> None:
        """Saving settings persists the vad_model_path to config."""
        recordings_path = tmp_path / "recordings"
        markdown_path = tmp_path / "markdown"
        vad_model_path = tmp_path / "custom_vad"
        config = Config(
            recordings_path=recordings_path,
            markdown_path=markdown_path,
            vad_model_path=vad_model_path,
        )
        status = Mock()
        selector_map = self._build_selector_map(recordings_path, markdown_path, status, vad_model_path)
        app = self._build_app(config, selector_map)
        handler = SettingsHandler(app)

        with (
            patch("utilityhub_config.get_config_path", return_value=tmp_path / "voicepad.yaml"),
            patch("utilityhub_config.write_config") as mock_write_config,
        ):
            handler.on_settings_save()

        # Verify vad_model_path directory was created
        assert vad_model_path.exists()

        # Verify write_config was called with correct config
        mock_write_config.assert_called_once()
        saved_config = mock_write_config.call_args[0][0]
        assert saved_config.vad_model_path == vad_model_path

    def test_vad_model_path_directory_creation_fails_gracefully(self, tmp_path: Path) -> None:
        """Saving settings handles vad_model_path directory creation failure."""
        recordings_path = tmp_path / "recordings"
        markdown_path = tmp_path / "markdown"
        vad_model_path = tmp_path / "blocked_vad"
        # Create a file where directory should be
        vad_model_path.write_text("blocking file", encoding="utf-8")

        config = Config(
            recordings_path=recordings_path,
            markdown_path=markdown_path,
            vad_model_path=vad_model_path,
        )
        status = Mock()
        selector_map = self._build_selector_map(recordings_path, markdown_path, status, vad_model_path)
        app = self._build_app(config, selector_map)
        handler = SettingsHandler(app)

        with (
            patch("utilityhub_config.get_config_path", return_value=tmp_path / "voicepad.yaml"),
            patch("utilityhub_config.write_config") as mock_write_config,
        ):
            handler.on_settings_save()

        # Should not write config if directory creation fails
        mock_write_config.assert_not_called()
        status.update.assert_called()

    def test_model_change_deactivates_runtime_before_warming(self, tmp_path: Path) -> None:
        """Saving a different model closes the active runtime before warming the replacement."""
        recordings_path = tmp_path / "recordings"
        markdown_path = tmp_path / "markdown"
        vad_model_path = tmp_path / "vad"
        config = Config(
            recordings_path=recordings_path,
            markdown_path=markdown_path,
            vad_model_path=vad_model_path,
            transcription_model="base",
        )
        status = Mock()
        selector_map = self._build_selector_map(recordings_path, markdown_path, status, vad_model_path)
        selector_map["#header-model"] = Mock()
        app = self._build_app(config, selector_map)
        handler = SettingsHandler(app)

        with (
            patch("utilityhub_config.get_config_path", return_value=tmp_path / "voicepad.yaml"),
            patch("utilityhub_config.write_config"),
            patch("voicepad_core.deactivate_model") as mock_deactivate,
        ):
            handler.on_settings_save()

        mock_deactivate.assert_called_once_with()
        app._warm_model_worker.assert_called_once_with()
