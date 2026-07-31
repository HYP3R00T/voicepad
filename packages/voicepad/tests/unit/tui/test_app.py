"""Tests for VoicePad TUI application."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
from voicepad_core.config import Config


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock Config instance."""
    config = Mock(spec=Config)
    config.recordings_path = Path("/tmp/recordings")
    config.markdown_path = Path("/tmp/markdown")
    config.vad_model_path = Path("/tmp/vad")
    config.logs_path = Path("/tmp/logs")
    config.log_level = "INFO"
    config.global_hotkey = "ctrl+shift+r"
    config.transcription_model = "base"
    config.transcription_device = "cpu"
    config.transcription_compute_type = "int8"
    config.input_device_index = None
    return config


class TestVoicePadAppInit:
    """Test VoicePadApp initialization."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_init_stores_config(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that __init__ stores the config."""
        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)
        assert app.config is mock_config

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_init_creates_managers(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that __init__ creates all managers."""
        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)
        assert app._layout_builder is not None
        assert app._lifecycle_manager is not None
        assert app._model_manager is not None
        assert app._timer_manager is not None
        assert app._tab_manager is not None


class TestRetranscribeFile:
    """Tests for history re-transcription."""

    def test_passes_live_config_to_transcription(self, tmp_path: Path) -> None:
        """Re-transcription uses the active TUI config, including vocabulary hints."""
        from voicepad.tui.app import VoicePadApp

        wav_path = tmp_path / "recording.wav"
        wav_path.touch()
        config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
            logs_path=tmp_path / "logs",
            proper_nouns=("Mise",),
        )
        app = Mock(spec=VoicePadApp)
        app.config = config
        app.call_from_thread = Mock()
        app._set_status = Mock()
        app._history_handler = Mock()
        result = Mock(text="Mise", latency_ms=1.0)

        with (
            patch("soundfile.read", return_value=(np.ones(16_000, dtype=np.float32), 16_000)),
            patch("voicepad_core.transcribe", return_value=result) as mock_transcribe,
            patch("voicepad.tui.app.begin_transcription_session", return_value=(Mock(), tmp_path / "log")),
            patch("voicepad.tui.app.end_transcription_session"),
            patch("voicepad.tui.app.log_transcription_start"),
            patch("voicepad.tui.app.log_transcription_end"),
        ):
            retranscribe = cast(
                Callable[[VoicePadApp, Path, Path | None], None],
                inspect.unwrap(VoicePadApp._retranscribe_file),
            )
            retranscribe(cast(VoicePadApp, app), wav_path, None)

        assert mock_transcribe.call_args.kwargs["config"] is config

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_init_creates_handlers(
        self,
        mock_hotkey_handler_class: Mock,
        mock_history_handler_class: Mock,
        mock_recording_handler_class: Mock,
        mock_settings_handler_class: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that __init__ creates all handlers."""
        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        mock_settings_handler_class.assert_called_once_with(app)
        mock_recording_handler_class.assert_called_once_with(app)
        mock_history_handler_class.assert_called_once_with(app)
        mock_hotkey_handler_class.assert_called_once_with(app)

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_init_sets_reactive_defaults(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that __init__ sets reactive attribute defaults."""
        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)
        assert app._model_ready is False
        assert app._recording is False
        assert app._transcribing is False


class TestLayoutMethods:
    """Test layout-related methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_compose_delegates_to_layout_builder(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that compose delegates to layout builder."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app._layout_builder, "compose", return_value=iter([])) as mock_compose:
            list(app.compose())
            mock_compose.assert_called_once()


class TestHotkeyDelegation:
    """Test hotkey-related delegation methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_start_hotkey_listener_delegates(
        self,
        mock_hotkey_handler_class: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _start_hotkey_listener delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_hotkey_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app._start_hotkey_listener()

        mock_handler.start_hotkey_listener.assert_called_once()

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_overlay_set_delegates(
        self,
        mock_hotkey_handler_class: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _overlay_set delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_hotkey_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app._overlay_set("recording")

        mock_handler.overlay_set.assert_called_once_with("recording")


class TestSettingsDelegation:
    """Test settings-related delegation methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_populate_settings_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler_class: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _populate_settings delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_settings_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app._populate_settings()

        mock_handler.populate_settings.assert_called_once()

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_get_hotkey_from_picker_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler_class: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _get_hotkey_from_picker delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_handler.get_hotkey_from_picker.return_value = "ctrl+shift+r"
        mock_settings_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        result = app._get_hotkey_from_picker()

        assert result == "ctrl+shift+r"
        mock_handler.get_hotkey_from_picker.assert_called_once()


class TestRecordingDelegation:
    """Test recording-related delegation methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_action_toggle_recording_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler_class: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that action_toggle_recording delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_recording_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app.action_toggle_recording()

        mock_handler.action_toggle_recording.assert_called_once()

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_start_recording_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler_class: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _start_recording delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_recording_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app._start_recording()

        mock_handler.start_recording.assert_called_once()


class TestHistoryDelegation:
    """Test history-related delegation methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_load_history_from_disk_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler_class: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _load_history_from_disk delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_history_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app._load_history_from_disk()

        mock_handler.load_history_from_disk.assert_called_once()

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_action_delete_entry_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler_class: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that action_delete_entry delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_history_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app.action_delete_entry()

        mock_handler.action_delete_entry.assert_called_once()

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_action_copy_transcription_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler_class: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that action_copy_transcription delegates to handler."""
        from voicepad.tui.app import VoicePadApp

        mock_handler = Mock()
        mock_history_handler_class.return_value = mock_handler
        app = VoicePadApp(mock_config)

        app.action_copy_transcription()

        mock_handler.action_copy_transcription.assert_called_once()


class TestModelManagerDelegation:
    """Test model manager delegation methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_warm_model_worker_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _warm_model_worker delegates to implementation."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app, "_warm_model_worker_impl") as mock_warm:
            app._warm_model_worker()
            mock_warm.assert_called_once()

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_set_status_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _set_status delegates to manager."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app._model_manager, "set_status") as mock_set_status:
            app._set_status("ready", "Model loaded")
            mock_set_status.assert_called_once_with("ready", "Model loaded")

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_action_reload_model_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that action_reload_model delegates to manager."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app._model_manager, "reload_model") as mock_reload:
            app.action_reload_model()
            mock_reload.assert_called_once()


class TestTimerManagerDelegation:
    """Test timer manager delegation methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_start_timer_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _start_timer delegates to manager."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app._timer_manager, "start_timer") as mock_start:
            app._start_timer()
            mock_start.assert_called_once()

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_stop_timer_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that _stop_timer delegates to manager."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app._timer_manager, "stop_timer") as mock_stop:
            app._stop_timer()
            mock_stop.assert_called_once()


class TestTabManagerDelegation:
    """Test tab manager delegation methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_check_action_delegates(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that check_action delegates to manager."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app._tab_manager, "check_action", return_value=True) as mock_check:
            result = app.check_action("toggle_recording", ())
            assert result is True
            mock_check.assert_called_once_with("toggle_recording", ())


class TestActionMethods:
    """Test action methods."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    def test_action_show_info_pushes_modal(
        self,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that action_show_info pushes InfoModal."""
        from unittest.mock import patch

        from voicepad.tui.app import VoicePadApp

        app = VoicePadApp(mock_config)

        with patch.object(app, "push_screen") as mock_push:
            app.action_show_info()
            mock_push.assert_called_once()
            # Check that the argument is an InfoModal instance
            args = mock_push.call_args[0]
            assert len(args) == 1
            from voicepad.tui.modals import InfoModal

            assert isinstance(args[0], InfoModal)


class TestActionOpenConfigDir:
    """Test action_open_config_dir method."""

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    @patch("platform.system")
    @patch("subprocess.run")
    @patch("utilityhub_config.get_config_path")
    def test_opens_config_dir_on_windows(
        self,
        mock_get_config_path: Mock,
        mock_subprocess_run: Mock,
        mock_platform_system: Mock,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
        tmp_path: Path,
    ) -> None:
        """Test that action_open_config_dir opens directory on Windows."""
        from voicepad.tui.app import VoicePadApp

        config_dir = tmp_path / "config"
        config_file = config_dir / "voicepad.yaml"
        mock_get_config_path.return_value = config_file
        mock_platform_system.return_value = "Windows"

        app = VoicePadApp(mock_config)
        app.action_open_config_dir()

        # Verify directory was created
        assert config_dir.exists()
        # Verify explorer was called with correct path
        mock_subprocess_run.assert_called_once_with(["explorer", str(config_dir)], check=False)

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    @patch("platform.system")
    @patch("subprocess.run")
    @patch("utilityhub_config.get_config_path")
    def test_opens_config_dir_on_macos(
        self,
        mock_get_config_path: Mock,
        mock_subprocess_run: Mock,
        mock_platform_system: Mock,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
        tmp_path: Path,
    ) -> None:
        """Test that action_open_config_dir opens directory on macOS."""
        from voicepad.tui.app import VoicePadApp

        config_dir = tmp_path / "config"
        config_file = config_dir / "voicepad.yaml"
        mock_get_config_path.return_value = config_file
        mock_platform_system.return_value = "Darwin"

        app = VoicePadApp(mock_config)
        app.action_open_config_dir()

        # Verify directory was created
        assert config_dir.exists()
        # Verify open was called with correct path
        mock_subprocess_run.assert_called_once_with(["open", str(config_dir)], check=False)

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    @patch("platform.system")
    @patch("subprocess.run")
    @patch("utilityhub_config.get_config_path")
    def test_opens_config_dir_on_linux(
        self,
        mock_get_config_path: Mock,
        mock_subprocess_run: Mock,
        mock_platform_system: Mock,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
        tmp_path: Path,
    ) -> None:
        """Test that action_open_config_dir opens directory on Linux."""
        from voicepad.tui.app import VoicePadApp

        config_dir = tmp_path / "config"
        config_file = config_dir / "voicepad.yaml"
        mock_get_config_path.return_value = config_file
        mock_platform_system.return_value = "Linux"

        app = VoicePadApp(mock_config)
        app.action_open_config_dir()

        # Verify directory was created
        assert config_dir.exists()
        # Verify xdg-open was called with correct path
        mock_subprocess_run.assert_called_once_with(["xdg-open", str(config_dir)], check=False)

    @patch("voicepad.tui.handlers.settings_handler.SettingsHandler")
    @patch("voicepad.tui.handlers.recording_handler.RecordingHandler")
    @patch("voicepad.tui.handlers.history_handler.HistoryHandler")
    @patch("voicepad.tui.handlers.hotkey_handler.HotkeyHandler")
    @patch("utilityhub_config.get_config_path")
    def test_handles_exception_gracefully(
        self,
        mock_get_config_path: Mock,
        mock_hotkey_handler: Mock,
        mock_history_handler: Mock,
        mock_recording_handler: Mock,
        mock_settings_handler: Mock,
        mock_config: Mock,
    ) -> None:
        """Test that action_open_config_dir handles exceptions gracefully."""
        from voicepad.tui.app import VoicePadApp

        # Make get_config_path raise an exception
        mock_get_config_path.side_effect = Exception("Config path error")

        app = VoicePadApp(mock_config)

        # Should not raise - exception is caught internally
        app.action_open_config_dir()


class TestRunFunction:
    """Test the run() entry point function."""

    @patch("voicepad.tui.app.get_config")
    @patch("voicepad.tui.app.configure_global_logging")
    @patch("voicepad.tui.app.VoicePadApp")
    def test_run_creates_app_and_runs(
        self, mock_app_class: Mock, mock_configure_logging: Mock, mock_get_config: Mock
    ) -> None:
        """Test that run() creates app with config and runs it."""
        from voicepad.tui.app import run

        mock_config = Mock()
        mock_get_config.return_value = mock_config
        mock_app = Mock()
        mock_app_class.return_value = mock_app
        mock_configure_logging.return_value = Path("/tmp/logs/session.log")

        run()

        mock_get_config.assert_called_once()
        mock_configure_logging.assert_called_once_with(mock_config.log_level, mock_config.logs_path, console=False)
        mock_app_class.assert_called_once_with(mock_config)
        mock_app.run.assert_called_once()
