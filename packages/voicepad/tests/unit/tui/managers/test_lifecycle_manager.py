"""Tests for LifecycleManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestLifecycleManagerInit:
    """Tests for LifecycleManager initialization."""

    def test_init_stores_app_reference(self) -> None:
        """Test that __init__ stores the app reference."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_app = Mock()
        manager = LifecycleManager(mock_app)

        assert manager.app is mock_app


class TestOnMount:
    """Tests for on_mount method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app.tui_config = Mock()
        app.tui_config.theme = "tokyo-night"
        app._load_history_from_disk = Mock()
        app._populate_settings = Mock()
        app._control_server = Mock()
        return app

    def test_registers_theme(self, mock_app: Mock) -> None:
        """Test that on_mount applies the theme from tui_config."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_app.tui_config = Mock()
        mock_app.tui_config.theme = "tokyo-night"
        manager = LifecycleManager(mock_app)

        with patch.object(manager, "check_first_run"):
            manager.on_mount()

            assert mock_app.theme == "tokyo-night"

    def test_sets_theme(self, mock_app: Mock) -> None:
        """Test that on_mount sets the app theme from tui_config."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_app.tui_config = Mock()
        mock_app.tui_config.theme = "dracula"
        manager = LifecycleManager(mock_app)

        with patch.object(manager, "check_first_run"):
            manager.on_mount()

            assert mock_app.theme == "dracula"

    def test_loads_history_from_disk(self, mock_app: Mock) -> None:
        """Test that on_mount loads history from disk."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with patch.object(manager, "check_first_run"):
            manager.on_mount()

            mock_app._load_history_from_disk.assert_called_once()

    def test_populates_settings(self, mock_app: Mock) -> None:
        """Test that on_mount populates settings."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with patch.object(manager, "check_first_run"):
            manager.on_mount()

            mock_app._populate_settings.assert_called_once()

    def test_checks_first_run(self, mock_app: Mock) -> None:
        """Test that on_mount checks for first run."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with patch.object(manager, "check_first_run") as mock_check:
            manager.on_mount()

            mock_check.assert_called_once()

    def test_starts_control_server(self, mock_app: Mock) -> None:
        """Mounting starts the local desktop-shortcut control channel."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with patch.object(manager, "check_first_run"):
            manager.on_mount()

        mock_app._control_server.start.assert_called_once_with()


class TestOnUnmount:
    """Tests for on_unmount method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app._hotkey_listener = None
        app._overlay = None
        app._control_server = Mock()
        return app

    def test_stops_hotkey_listener_if_running(self, mock_app: Mock) -> None:
        """Test that on_unmount stops the hotkey listener if running."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_listener = Mock()
        mock_app._hotkey_listener = mock_listener
        manager = LifecycleManager(mock_app)

        manager.on_unmount()

        mock_listener.stop.assert_called_once()

    def test_does_not_stop_hotkey_listener_if_not_running(self, mock_app: Mock) -> None:
        """Test that on_unmount handles None hotkey listener."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_app._hotkey_listener = None
        manager = LifecycleManager(mock_app)

        # Should not raise an exception
        manager.on_unmount()

    def test_stops_overlay_if_running(self, mock_app: Mock) -> None:
        """Test that on_unmount stops the overlay if running."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_overlay = Mock()
        mock_app._overlay = mock_overlay
        manager = LifecycleManager(mock_app)

        manager.on_unmount()

        mock_overlay.stop.assert_called_once()

    def test_stops_control_server(self, mock_app: Mock) -> None:
        """Unmounting stops the local desktop-shortcut control channel."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        manager.on_unmount()

        mock_app._control_server.stop.assert_called_once_with()

    def test_does_not_stop_overlay_if_not_running(self, mock_app: Mock) -> None:
        """Test that on_unmount handles None overlay."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_app._overlay = None
        manager = LifecycleManager(mock_app)

        # Should not raise an exception
        manager.on_unmount()

    def test_handles_hotkey_listener_exception(self, mock_app: Mock) -> None:
        """Test that on_unmount handles exceptions when stopping hotkey listener."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_listener = Mock()
        mock_listener.stop.side_effect = RuntimeError("Stop failed")
        mock_app._hotkey_listener = mock_listener
        manager = LifecycleManager(mock_app)

        # Should not raise an exception
        manager.on_unmount()

    def test_handles_overlay_exception(self, mock_app: Mock) -> None:
        """Test that on_unmount handles exceptions when stopping overlay."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        mock_overlay = Mock()
        mock_overlay.stop.side_effect = RuntimeError("Stop failed")
        mock_app._overlay = mock_overlay
        manager = LifecycleManager(mock_app)

        # Should not raise an exception
        manager.on_unmount()

    def test_deactivates_model_runtime(self, mock_app: Mock) -> None:
        """Unmounting releases the active model runtime."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with patch("voicepad_core.deactivate_model") as mock_deactivate:
            manager.on_unmount()

        mock_deactivate.assert_called_once_with()

    def test_logs_model_deactivation_failure(self, mock_app: Mock, caplog: pytest.LogCaptureFixture) -> None:
        """A model shutdown failure is logged without blocking app shutdown."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with patch("voicepad_core.deactivate_model", side_effect=RuntimeError("close failed")):
            manager.on_unmount()

        assert "Could not deactivate the transcription model during shutdown: close failed" in caplog.text


class TestCheckFirstRun:
    """Tests for check_first_run method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app.config = Mock()
        app.config.transcription_model = "base"
        app.push_screen = Mock()
        app._warm_model_worker = Mock()
        return app

    def test_shows_setup_modal_when_config_missing(self, mock_app: Mock) -> None:
        """Test that check_first_run shows setup modal when config is missing."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("voicepad_core.model_is_ready") as mock_ready,
        ):
            mock_path = Mock(spec=Path)
            mock_path.exists.return_value = False
            mock_get_path.return_value = mock_path
            mock_ready.return_value = True

            manager.check_first_run()

            mock_app.push_screen.assert_called_once()
            mock_app._warm_model_worker.assert_not_called()

    def test_shows_setup_modal_when_model_not_downloaded(self, mock_app: Mock) -> None:
        """Test that check_first_run shows setup modal when model not downloaded."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("voicepad_core.model_is_ready") as mock_ready,
        ):
            mock_path = Mock(spec=Path)
            mock_path.exists.return_value = True
            mock_get_path.return_value = mock_path
            mock_ready.return_value = False

            manager.check_first_run()

            mock_app.push_screen.assert_called_once()
            mock_app._warm_model_worker.assert_not_called()

    def test_warms_model_when_config_and_model_ready(self, mock_app: Mock) -> None:
        """Test that check_first_run warms model when config and model are ready."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("voicepad_core.model_is_ready") as mock_ready,
        ):
            mock_path = Mock(spec=Path)
            mock_path.exists.return_value = True
            mock_get_path.return_value = mock_path
            mock_ready.return_value = True

            manager.check_first_run()

            mock_app.push_screen.assert_not_called()
            mock_app._warm_model_worker.assert_called_once()

    def test_setup_modal_callback_calls_on_setup_done(self, mock_app: Mock) -> None:
        """Test that setup modal callback calls on_setup_done with result."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("voicepad_core.model_is_ready") as mock_ready,
            patch.object(manager, "on_setup_done") as mock_on_setup,
        ):
            mock_path = Mock(spec=Path)
            mock_path.exists.return_value = False
            mock_get_path.return_value = mock_path
            mock_ready.return_value = True

            manager.check_first_run()

            # Get the callback function
            callback = mock_app.push_screen.call_args[1]["callback"]
            result = ("base", 0)
            callback(result)

            mock_on_setup.assert_called_once_with(result)

    def test_setup_modal_callback_handles_none_result(self, mock_app: Mock) -> None:
        """Test that setup modal callback handles None result."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("voicepad_core.model_is_ready") as mock_ready,
            patch.object(manager, "on_setup_done") as mock_on_setup,
        ):
            mock_path = Mock(spec=Path)
            mock_path.exists.return_value = False
            mock_get_path.return_value = mock_path
            mock_ready.return_value = True

            manager.check_first_run()

            # Get the callback function
            callback = mock_app.push_screen.call_args[1]["callback"]
            callback(None)

            mock_on_setup.assert_not_called()


class TestOnSetupDone:
    """Tests for on_setup_done method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app.config = Mock()
        app.config.model_dump.return_value = {
            "transcription_model": "old_model",
            "input_device_index": None,
        }
        app._refresh_config_path_label = Mock()
        app._refresh_settings_values = Mock()
        app._warm_model_worker = Mock()
        return app

    def test_updates_config_with_chosen_model(self, mock_app: Mock) -> None:
        """Test that on_setup_done updates config with chosen model."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config"),
            patch("voicepad_core.config.Config") as mock_config_class,
        ):
            mock_path = Mock(spec=Path)
            mock_path.parent = Mock()
            mock_get_path.return_value = mock_path
            mock_new_config = Mock()
            mock_config_class.return_value = mock_new_config

            manager.on_setup_done(result)

            # Check that Config was called with updated values
            call_args = mock_config_class.call_args[1]
            assert call_args["transcription_model"] == "base"
            assert call_args["input_device_index"] == 0

    def test_writes_config_to_disk(self, mock_app: Mock) -> None:
        """Test that on_setup_done writes config to disk."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config") as mock_write,
            patch("voicepad_core.config.Config") as mock_config_class,
        ):
            mock_path = Mock(spec=Path)
            mock_path.parent = Mock()
            mock_get_path.return_value = mock_path
            mock_new_config = Mock()
            mock_config_class.return_value = mock_new_config

            manager.on_setup_done(result)

            mock_write.assert_called_once_with(mock_new_config, "voicepad", path=mock_path, format="yaml")

    def test_creates_config_directory(self, mock_app: Mock) -> None:
        """Test that on_setup_done creates config directory if needed."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config"),
            patch("voicepad_core.config.Config"),
        ):
            mock_path = Mock(spec=Path)
            mock_parent = Mock()
            mock_path.parent = mock_parent
            mock_get_path.return_value = mock_path

            manager.on_setup_done(result)

            mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_updates_app_config(self, mock_app: Mock) -> None:
        """Test that on_setup_done updates the app config."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config"),
            patch("voicepad_core.config.Config") as mock_config_class,
        ):
            mock_path = Mock(spec=Path)
            mock_path.parent = Mock()
            mock_get_path.return_value = mock_path
            mock_new_config = Mock()
            mock_config_class.return_value = mock_new_config

            manager.on_setup_done(result)

            # Check that app.config was updated
            assert mock_app.config is mock_new_config

    def test_refreshes_config_path_label(self, mock_app: Mock) -> None:
        """Test that on_setup_done refreshes config path label."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config"),
            patch("voicepad_core.config.Config"),
        ):
            mock_path = Mock(spec=Path)
            mock_path.parent = Mock()
            mock_get_path.return_value = mock_path

            manager.on_setup_done(result)

            mock_app._refresh_config_path_label.assert_called_once()

    def test_refreshes_settings_values(self, mock_app: Mock) -> None:
        """Test that on_setup_done refreshes settings values."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config"),
            patch("voicepad_core.config.Config"),
        ):
            mock_path = Mock(spec=Path)
            mock_path.parent = Mock()
            mock_get_path.return_value = mock_path

            manager.on_setup_done(result)

            mock_app._refresh_settings_values.assert_called_once()

    def test_warms_model_after_setup(self, mock_app: Mock) -> None:
        """Test that on_setup_done warms model after setup."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config"),
            patch("voicepad_core.config.Config"),
        ):
            mock_path = Mock(spec=Path)
            mock_path.parent = Mock()
            mock_get_path.return_value = mock_path

            manager.on_setup_done(result)

            mock_app._warm_model_worker.assert_called_once()

    def test_handles_config_write_exception(self, mock_app: Mock) -> None:
        """Test that on_setup_done handles exceptions when writing config."""
        from voicepad.tui.managers.lifecycle_manager import LifecycleManager

        manager = LifecycleManager(mock_app)
        result = ("base", 0)

        with (
            patch("utilityhub_config.get_config_path") as mock_get_path,
            patch("utilityhub_config.write_config") as mock_write,
            patch("voicepad_core.config.Config"),
        ):
            mock_path = Mock(spec=Path)
            mock_path.parent = Mock()
            mock_get_path.return_value = mock_path
            mock_write.side_effect = RuntimeError("Write failed")

            # Should not raise an exception
            manager.on_setup_done(result)

            # Should still warm model
            mock_app._warm_model_worker.assert_called_once()
