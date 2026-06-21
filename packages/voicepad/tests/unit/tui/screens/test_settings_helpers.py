"""Tests for settings_helpers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.app import App
from textual.widgets import Checkbox, Input, Label, Select, Static
from voicepad.tui.screens.settings_helpers import populate_settings_form

# Patch target for _get_input_devices (imported inside populate_settings_form)
PATCH_TARGET = "voicepad.cli.config._get_input_devices"


class SettingsTestApp(App[None]):
    """Test app for settings helpers."""

    def __init__(self, config: MagicMock, config_path: str) -> None:
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.container: Static | None = None

    def on_mount(self) -> None:
        self.container = Static(id="settings-container")
        self.mount(self.container)
        # Call populate_settings_form during mount, before widgets are fully mounted
        populate_settings_form(self.container, self.config, self.config_path)


def create_mock_config() -> MagicMock:
    """Create a mock config object with model_fields."""
    config = MagicMock()
    config.recordings_path = "/path/to/recordings"
    config.markdown_path = "/path/to/markdown"
    config.transcription_model = "turbo"
    config.input_device_index = -1
    config.global_hotkey = "ctrl+alt+v"

    # Mock the model_fields class attribute
    mock_field = MagicMock()
    mock_field.default = "default_value"
    type(config).model_fields = {
        "recordings_path": mock_field,
        "markdown_path": mock_field,
        "transcription_model": mock_field,
        "input_device_index": mock_field,
    }

    return config


def create_mock_devices() -> list[MagicMock]:
    """Create mock audio devices."""
    device1 = MagicMock()
    device1.index = 0
    device1.name = "Microphone 1"

    device2 = MagicMock()
    device2.index = 1
    device2.name = "Microphone 2"

    return [device1, device2]


class TestPopulateSettingsForm:
    """Test suite for populate_settings_form function."""

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_mounts_config_path_label(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form mounts a label showing the config path."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config_path = "/path/to/config.yaml"

        app = SettingsTestApp(config, config_path)
        async with app.run_test():
            assert app.container is not None

            label = app.container.query_one("#settings-config-path", Label)
            assert config_path in str(label.render())

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_config_path_shows_not_created_hint_when_missing(self, mock_get_devices: MagicMock) -> None:
        """Config path label shows '(not yet created)' when file doesn't exist."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config_path = "/nonexistent/config.yaml"

        app = SettingsTestApp(config, config_path)
        async with app.run_test():
            assert app.container is not None

            label = app.container.query_one("#settings-config-path", Label)
            assert "(not yet created)" in str(label.render())

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_creates_input_for_recordings_path(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form creates an Input widget for recordings_path."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            input_widget = app.container.query_one("#setting-recordings_path", Input)
            assert input_widget.value == "/path/to/recordings"

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_creates_input_for_markdown_path(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form creates an Input widget for markdown_path."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            input_widget = app.container.query_one("#setting-markdown_path", Input)
            assert input_widget.value == "/path/to/markdown"

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_creates_select_for_transcription_model(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form creates a Select widget for transcription_model."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            select = app.container.query_one("#setting-transcription_model", Select)
            assert select.value == "turbo"
            # Check that valid models are in options
            option_values = [opt[1] for opt in select._options]
            assert "turbo" in option_values
            assert option_values == ["small", "large-v3", "turbo"]

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_creates_select_for_input_device(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form creates a Select widget for input_device_index."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            select = app.container.query_one("#setting-input_device_index", Select)
            assert select.value == -1

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_device_select_includes_system_default(self, mock_get_devices: MagicMock) -> None:
        """Device select includes 'system default' option with value -1."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            select = app.container.query_one("#setting-input_device_index", Select)
            options = select._options
            # First option should be system default
            assert options[0][0] == "System default"
            assert options[0][1] == -1

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_device_select_includes_all_devices(self, mock_get_devices: MagicMock) -> None:
        """Device select includes all devices from _get_input_devices."""
        devices = create_mock_devices()
        mock_get_devices.return_value = devices
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            select = app.container.query_one("#setting-input_device_index", Select)
            option_values = [opt[1] for opt in select._options]
            assert 0 in option_values
            assert 1 in option_values

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_creates_settings_rows(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form creates settings-row containers."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            rows = app.container.query(".settings-row")
            # 4 config fields + 1 hotkey row
            assert len(rows) >= 5

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_creates_key_labels_with_hints(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form creates labels with field names and hints."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            labels = app.container.query(".settings-key")
            assert len(labels) >= 4
            # Check that at least one label contains a field name
            label_texts = [str(label.render()) for label in labels]
            assert any("recordings_path" in text for text in label_texts)

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_mounts_hotkey_picker(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form mounts the hotkey picker."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            hotkey_row = app.container.query_one("#hotkey-row", Static)
            assert hotkey_row is not None

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_hotkey_picker_has_modifier_checkboxes(self, mock_get_devices: MagicMock) -> None:
        """Hotkey picker includes checkboxes for Ctrl, Alt, Shift, Win."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            ctrl_cb = app.container.query_one("#hotkey-mod-ctrl", Checkbox)
            alt_cb = app.container.query_one("#hotkey-mod-alt", Checkbox)
            shift_cb = app.container.query_one("#hotkey-mod-shift", Checkbox)
            win_cb = app.container.query_one("#hotkey-mod-cmd", Checkbox)

            assert ctrl_cb is not None
            assert alt_cb is not None
            assert shift_cb is not None
            assert win_cb is not None

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_hotkey_picker_parses_config_hotkey(self, mock_get_devices: MagicMock) -> None:
        """Hotkey picker parses the config hotkey and checks appropriate modifiers."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.global_hotkey = "ctrl+alt+v"

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            ctrl_cb = app.container.query_one("#hotkey-mod-ctrl", Checkbox)
            alt_cb = app.container.query_one("#hotkey-mod-alt", Checkbox)
            shift_cb = app.container.query_one("#hotkey-mod-shift", Checkbox)

            assert ctrl_cb.value is True
            assert alt_cb.value is True
            assert shift_cb.value is False

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_hotkey_picker_has_key_select(self, mock_get_devices: MagicMock) -> None:
        """Hotkey picker includes a Select widget for the key."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            key_select = app.container.query_one("#hotkey-key-select", Select)
            assert key_select is not None

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_hotkey_picker_has_preview_label(self, mock_get_devices: MagicMock) -> None:
        """Hotkey picker includes a preview label."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            preview = app.container.query_one("#hotkey-preview", Label)
            assert preview is not None

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_handles_none_values_in_config(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form handles None values in config gracefully."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.recordings_path = None
        config.markdown_path = None

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            recordings_input = app.container.query_one("#setting-recordings_path", Input)
            markdown_input = app.container.query_one("#setting-markdown_path", Input)
            assert recordings_input.value == ""
            assert markdown_input.value == ""

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_uses_default_model_when_invalid(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form uses 'turbo' when model is invalid."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.transcription_model = "invalid-model"

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            select = app.container.query_one("#setting-transcription_model", Select)
            assert select.value == "turbo"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_preserves_current_advanced_model_when_configured(self, mock_get_devices: MagicMock) -> None:
        """Advanced config models remain selectable even though the default UI is curated."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.transcription_model = "base.en"

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            select = app.container.query_one("#setting-transcription_model", Select)
            assert select.value == "base.en"
            option_values = [opt[1] for opt in select._options]
            assert option_values == ["small", "large-v3", "turbo", "base.en"]

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_uses_default_device_when_invalid(self, mock_get_devices: MagicMock) -> None:
        """populate_settings_form uses -1 when device index is invalid."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.input_device_index = 999  # Invalid device index

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            select = app.container.query_one("#setting-input_device_index", Select)
            assert select.value == -1

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_input_widgets_have_placeholders(self, mock_get_devices: MagicMock) -> None:
        """Input widgets show placeholder text from field defaults."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            recordings_input = app.container.query_one("#setting-recordings_path", Input)
            assert recordings_input.placeholder == "default_value"

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_select_widgets_exist(self, mock_get_devices: MagicMock) -> None:
        """Select widgets are created for model and device."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            model_select = app.container.query_one("#setting-transcription_model", Select)
            device_select = app.container.query_one("#setting-input_device_index", Select)
            assert model_select is not None
            assert device_select is not None

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_hotkey_mod_row_exists(self, mock_get_devices: MagicMock) -> None:
        """Hotkey modifier row is created."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            mod_row = app.container.query_one("#hotkey-mod-row", Static)
            assert mod_row is not None

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_all_modifiers_have_correct_classes(self, mock_get_devices: MagicMock) -> None:
        """All modifier checkboxes have the hotkey-checkbox class."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            checkboxes = app.container.query(".hotkey-checkbox")
            assert len(checkboxes) == 4

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_hotkey_with_shift_modifier(self, mock_get_devices: MagicMock) -> None:
        """Hotkey picker correctly parses shift modifier."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.global_hotkey = "shift+s"

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            shift_cb = app.container.query_one("#hotkey-mod-shift", Checkbox)
            ctrl_cb = app.container.query_one("#hotkey-mod-ctrl", Checkbox)
            key_select = app.container.query_one("#hotkey-key-select", Select)

            assert shift_cb.value is True
            assert ctrl_cb.value is False
            assert key_select.value == "s"

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_hotkey_with_win_modifier(self, mock_get_devices: MagicMock) -> None:
        """Hotkey picker correctly parses win/cmd modifier."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.global_hotkey = "cmd+w"

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            win_cb = app.container.query_one("#hotkey-mod-cmd", Checkbox)
            key_select = app.container.query_one("#hotkey-key-select", Select)

            assert win_cb.value is True
            assert key_select.value == "w"

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_empty_hotkey_defaults_to_v(self, mock_get_devices: MagicMock) -> None:
        """Empty hotkey string defaults to 'v' key."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.global_hotkey = ""

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            key_select = app.container.query_one("#hotkey-key-select", Select)
            assert key_select.value == "v"

    @pytest.mark.asyncio
    @patch(PATCH_TARGET)
    async def test_invalid_hotkey_key_defaults_to_v(self, mock_get_devices: MagicMock) -> None:
        """Invalid hotkey key defaults to 'v'."""
        mock_get_devices.return_value = create_mock_devices()
        config = create_mock_config()
        config.global_hotkey = "ctrl+invalid_key"

        app = SettingsTestApp(config, "/path/to/config.yaml")
        async with app.run_test():
            assert app.container is not None

            key_select = app.container.query_one("#hotkey-key-select", Select)
            assert key_select.value == "v"
