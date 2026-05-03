"""Tests for SettingsService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from voicepad.tui.services.settings_service import SettingsService


def create_mock_config() -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.recordings_path = Path("/path/to/recordings")
    config.markdown_path = Path("/path/to/markdown")
    config.transcription_model = "turbo"
    config.input_device_index = -1
    config.global_hotkey = "ctrl+alt+v"
    config.model_dump.return_value = {
        "recordings_path": "/path/to/recordings",
        "markdown_path": "/path/to/markdown",
        "transcription_model": "turbo",
        "input_device_index": -1,
        "global_hotkey": "ctrl+alt+v",
    }
    return config


class TestSettingsService:
    """Test suite for SettingsService."""

    def test_init_stores_config(self) -> None:
        """SettingsService stores the config object."""
        config = create_mock_config()
        service = SettingsService(config)
        assert service.config == config

    @patch("voicepad.tui.services.settings_service.get_config_path")
    def test_get_config_path_returns_path(self, mock_get_path: MagicMock) -> None:
        """get_config_path returns the config file path."""
        config = create_mock_config()
        service = SettingsService(config)

        expected_path = Path("/home/user/.config/voicepad/config.yaml")
        mock_get_path.return_value = expected_path

        result = service.get_config_path()

        assert result == expected_path
        mock_get_path.assert_called_once_with("voicepad", format="yaml")

    @patch("voicepad.tui.services.settings_service.get_config_path")
    def test_config_exists_returns_true_when_file_exists(self, mock_get_path: MagicMock, tmp_path: Path) -> None:
        """config_exists returns True when config file exists."""
        config = create_mock_config()
        service = SettingsService(config)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")
        mock_get_path.return_value = config_file

        assert service.config_exists() is True

    @patch("voicepad.tui.services.settings_service.get_config_path")
    def test_config_exists_returns_false_when_file_missing(self, mock_get_path: MagicMock, tmp_path: Path) -> None:
        """config_exists returns False when config file doesn't exist."""
        config = create_mock_config()
        service = SettingsService(config)

        config_file = tmp_path / "nonexistent.yaml"
        mock_get_path.return_value = config_file

        assert service.config_exists() is False

    @patch("voicepad.tui.services.settings_service.write_config")
    @patch("voicepad.tui.services.settings_service.get_config_path")
    def test_save_config_creates_parent_directory(
        self, mock_get_path: MagicMock, mock_write: MagicMock, tmp_path: Path
    ) -> None:
        """save_config creates parent directory if it doesn't exist."""
        config = create_mock_config()
        service = SettingsService(config)

        config_file = tmp_path / "subdir" / "config.yaml"
        mock_get_path.return_value = config_file

        service.save_config(config)

        assert config_file.parent.exists()

    @patch("voicepad.tui.services.settings_service.write_config")
    @patch("voicepad.tui.services.settings_service.get_config_path")
    def test_save_config_calls_write_config(
        self, mock_get_path: MagicMock, mock_write: MagicMock, tmp_path: Path
    ) -> None:
        """save_config calls write_config with correct parameters."""
        config = create_mock_config()
        service = SettingsService(config)

        config_file = tmp_path / "config.yaml"
        mock_get_path.return_value = config_file

        service.save_config(config)

        mock_write.assert_called_once_with(config, "voicepad", path=config_file, format="yaml")

    @patch("voicepad.tui.services.settings_service.write_config")
    @patch("voicepad.tui.services.settings_service.get_config_path")
    def test_save_config_raises_on_write_error(
        self, mock_get_path: MagicMock, mock_write: MagicMock, tmp_path: Path
    ) -> None:
        """save_config raises exception if write fails."""
        config = create_mock_config()
        service = SettingsService(config)

        config_file = tmp_path / "config.yaml"
        mock_get_path.return_value = config_file
        mock_write.side_effect = Exception("Write failed")

        with pytest.raises(Exception, match="Write failed"):
            service.save_config(config)

    @patch("voicepad_core.config.Config")
    @patch("voicepad.tui.services.settings_service.SettingsService.save_config")
    def test_update_field_updates_single_field(self, mock_save: MagicMock, mock_config_class: MagicMock) -> None:
        """update_field updates a single configuration field."""
        config = create_mock_config()
        service = SettingsService(config)

        new_config = MagicMock()
        mock_config_class.return_value = new_config

        result = service.update_field("transcription_model", "base")

        # Verify the field was updated in the raw dict
        call_args = mock_config_class.call_args[1]
        assert call_args["transcription_model"] == "base"
        assert result == new_config
        mock_save.assert_called_once_with(new_config)

    @patch("voicepad_core.config.Config")
    @patch("voicepad.tui.services.settings_service.SettingsService.save_config")
    def test_update_field_preserves_other_fields(self, mock_save: MagicMock, mock_config_class: MagicMock) -> None:
        """update_field preserves other configuration fields."""
        config = create_mock_config()
        service = SettingsService(config)

        new_config = MagicMock()
        mock_config_class.return_value = new_config

        service.update_field("transcription_model", "base")

        # Verify other fields are preserved
        call_args = mock_config_class.call_args[1]
        assert call_args["recordings_path"] == "/path/to/recordings"
        assert call_args["markdown_path"] == "/path/to/markdown"

    @patch("voicepad_core.config.Config")
    @patch("voicepad.tui.services.settings_service.SettingsService.save_config")
    def test_update_field_raises_on_invalid_value(self, mock_save: MagicMock, mock_config_class: MagicMock) -> None:
        """update_field raises exception if value is invalid."""
        config = create_mock_config()
        service = SettingsService(config)

        mock_config_class.side_effect = ValueError("Invalid value")

        with pytest.raises(ValueError, match="Invalid value"):
            service.update_field("transcription_model", "invalid")

    @patch("voicepad_core.config.Config")
    @patch("voicepad.tui.services.settings_service.SettingsService.save_config")
    def test_update_fields_updates_multiple_fields(self, mock_save: MagicMock, mock_config_class: MagicMock) -> None:
        """update_fields updates multiple configuration fields."""
        config = create_mock_config()
        service = SettingsService(config)

        updates = {
            "transcription_model": "base",
            "input_device_index": 0,
        }

        new_config = MagicMock()
        mock_config_class.return_value = new_config

        result = service.update_fields(updates)

        # Verify all fields were updated
        call_args = mock_config_class.call_args[1]
        assert call_args["transcription_model"] == "base"
        assert call_args["input_device_index"] == 0
        assert result == new_config
        mock_save.assert_called_once_with(new_config)

    @patch("voicepad_core.config.Config")
    @patch("voicepad.tui.services.settings_service.SettingsService.save_config")
    def test_update_fields_preserves_unchanged_fields(self, mock_save: MagicMock, mock_config_class: MagicMock) -> None:
        """update_fields preserves fields not in the updates dict."""
        config = create_mock_config()
        service = SettingsService(config)

        updates = {"transcription_model": "base"}

        new_config = MagicMock()
        mock_config_class.return_value = new_config

        service.update_fields(updates)

        # Verify unchanged fields are preserved
        call_args = mock_config_class.call_args[1]
        assert call_args["recordings_path"] == "/path/to/recordings"
        assert call_args["global_hotkey"] == "ctrl+alt+v"

    @patch("voicepad_core.config.Config")
    @patch("voicepad.tui.services.settings_service.SettingsService.save_config")
    def test_update_fields_raises_on_invalid_values(self, mock_save: MagicMock, mock_config_class: MagicMock) -> None:
        """update_fields raises exception if any value is invalid."""
        config = create_mock_config()
        service = SettingsService(config)

        updates = {"transcription_model": "invalid"}

        mock_config_class.side_effect = ValueError("Invalid value")

        with pytest.raises(ValueError, match="Invalid value"):
            service.update_fields(updates)

    @patch("voicepad_core.config.Config")
    def test_validate_field_returns_true_for_valid_value(self, mock_config_class: MagicMock) -> None:
        """validate_field returns (True, None) for valid value."""
        config = create_mock_config()
        service = SettingsService(config)

        mock_config_class.return_value = MagicMock()

        is_valid, error = service.validate_field("transcription_model", "base")

        assert is_valid is True
        assert error is None

    @patch("voicepad_core.config.Config")
    def test_validate_field_returns_false_for_invalid_value(self, mock_config_class: MagicMock) -> None:
        """validate_field returns (False, error_message) for invalid value."""
        config = create_mock_config()
        service = SettingsService(config)

        mock_config_class.side_effect = ValueError("Invalid model")

        is_valid, error = service.validate_field("transcription_model", "invalid")

        assert is_valid is False
        assert error == "Invalid model"

    @patch("voicepad_core.config.Config")
    def test_validate_field_does_not_save_config(self, mock_config_class: MagicMock) -> None:
        """validate_field does not save the configuration."""
        config = create_mock_config()
        service = SettingsService(config)

        with patch.object(service, "save_config") as mock_save:
            mock_config_class.return_value = MagicMock()

            service.validate_field("transcription_model", "base")

            mock_save.assert_not_called()

    @patch("voicepad_core.config.Config")
    def test_validate_field_preserves_original_config(self, mock_config_class: MagicMock) -> None:
        """validate_field does not modify the original config."""
        config = create_mock_config()
        service = SettingsService(config)

        original_model = config.transcription_model

        mock_config_class.return_value = MagicMock()

        service.validate_field("transcription_model", "base")

        assert config.transcription_model == original_model
