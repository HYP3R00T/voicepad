"""Tests for settings handler."""

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
    ) -> dict[str, object]:
        recordings_input = Mock()
        recordings_input.value = str(recordings_path)
        markdown_input = Mock()
        markdown_input.value = str(markdown_path)

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
        config = Config(recordings_path=recordings_path, markdown_path=markdown_path)
        status = Mock()
        selector_map = self._build_selector_map(recordings_path, markdown_path, status)
        app = self._build_app(config, selector_map)
        handler = SettingsHandler(app)

        with (
            patch("utilityhub_config.get_config_path", return_value=tmp_path / "voicepad.yaml"),
            patch("utilityhub_config.write_config") as mock_write_config,
        ):
            handler.on_settings_save()

        assert recordings_path.exists()
        assert markdown_path.exists()
        mock_write_config.assert_called_once()
        status.update.assert_called()

    def test_reports_error_when_recordings_path_cannot_be_created(self, tmp_path: Path) -> None:
        """Saving settings shows an error if a directory path cannot be created."""
        recordings_path = tmp_path / "blocked"
        recordings_path.write_text("not a directory", encoding="utf-8")
        markdown_path = tmp_path / "notes" / "markdown"
        config = Config(recordings_path=recordings_path, markdown_path=markdown_path)
        status = Mock()
        selector_map = self._build_selector_map(recordings_path, markdown_path, status)
        app = self._build_app(config, selector_map)
        handler = SettingsHandler(app)

        with (
            patch("utilityhub_config.get_config_path", return_value=tmp_path / "voicepad.yaml"),
            patch("utilityhub_config.write_config") as mock_write_config,
        ):
            handler.on_settings_save()

        mock_write_config.assert_not_called()
        status.update.assert_called()
