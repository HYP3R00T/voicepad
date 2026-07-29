"""Tests for hotkey handler."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import ANY, Mock, patch

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def mock_app() -> Mock:
    """Create a mock VoicePadApp instance."""
    app = Mock(
        spec=[
            "config",
            "tui_config",
            "_overlay",
            "_hotkey_listener",
            "_recording_handler",
            "_recording",
            "_transcribing",
            "_model_ready",
            "_hotkey_pending_copy",
            "call_from_thread",
            "query_one",
        ]
    )
    app.config = Mock()
    app.config.global_hotkey = "ctrl+shift+r"
    app.tui_config = Mock()
    app.tui_config.theme = "tokyo-night"
    app._overlay = None
    app._hotkey_listener = None
    app._recording_handler = Mock()
    app._recording = False
    app._transcribing = False
    app._model_ready = True
    app._hotkey_pending_copy = False
    return app


class TestHotkeyHandlerInit:
    """Test HotkeyHandler initialization."""

    def test_init_stores_app_reference(self, mock_app: Mock) -> None:
        """Test that __init__ stores the app reference."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        handler = HotkeyHandler(mock_app)
        assert handler.app is mock_app


class TestStartHotkeyListener:
    """Test start_hotkey_listener method."""

    def test_returns_early_when_no_hotkey_configured(self, mock_app: Mock) -> None:
        """Windows returns early when no native hotkey is configured."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app.config.global_hotkey = ""
        handler = HotkeyHandler(mock_app)

        with patch("voicepad.tui.handlers.hotkey_handler.platform.system", return_value="Windows"):
            handler.start_hotkey_listener()

        assert mock_app._overlay is None
        assert mock_app._hotkey_listener is None

    @patch("voicepad.tui.overlay.StatusOverlay")
    def test_linux_empty_windows_hotkey_still_starts_overlay(self, mock_overlay_class: Mock, mock_app: Mock) -> None:
        """Linux control requests keep status feedback when the Windows setting is empty."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app.config.global_hotkey = ""
        handler = HotkeyHandler(mock_app)

        with patch("voicepad.tui.handlers.hotkey_handler.platform.system", return_value="Linux"):
            handler.start_hotkey_listener()

        mock_overlay_class.return_value.start.assert_called_once_with()

    @patch("voicepad.tui.hotkey.GlobalHotkeyListener")
    @patch("voicepad.tui.overlay.StatusOverlay")
    def test_creates_overlay_and_listener(
        self, mock_overlay_class: Mock, mock_listener_class: Mock, mock_app: Mock
    ) -> None:
        """Test that start_hotkey_listener creates overlay and listener."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_overlay = Mock()
        mock_overlay_class.return_value = mock_overlay
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener

        handler = HotkeyHandler(mock_app)
        handler.start_hotkey_listener()

        mock_overlay_class.assert_called_once()
        mock_overlay.start.assert_called_once()
        assert mock_app._overlay is mock_overlay

    @patch("voicepad.tui.hotkey.GlobalHotkeyListener")
    @patch("voicepad.tui.overlay.StatusOverlay")
    def test_starts_hotkey_listener_with_callbacks(
        self, mock_overlay_class: Mock, mock_listener_class: Mock, mock_app: Mock
    ) -> None:
        """Test that start_hotkey_listener starts listener with correct callbacks."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener

        handler = HotkeyHandler(mock_app)
        with patch("voicepad.tui.handlers.hotkey_handler.platform.system", return_value="Windows"):
            handler.start_hotkey_listener()

        mock_listener_class.assert_called_once_with(
            hotkey="ctrl+shift+r",
            on_toggle=handler.hotkey_on_toggle,
        )
        mock_listener.start.assert_called_once()
        assert mock_app._hotkey_listener is mock_listener

    @patch("voicepad.tui.hotkey.GlobalHotkeyListener")
    @patch("voicepad.tui.overlay.StatusOverlay")
    def test_linux_uses_desktop_shortcut_without_native_listener(
        self, mock_overlay_class: Mock, mock_listener_class: Mock, mock_app: Mock
    ) -> None:
        """Linux starts the overlay but leaves global registration to the desktop."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        handler = HotkeyHandler(mock_app)
        with patch("voicepad.tui.handlers.hotkey_handler.platform.system", return_value="Linux"):
            handler.start_hotkey_listener()

        mock_listener_class.assert_not_called()
        assert mock_app._hotkey_listener is None

    @patch("voicepad.tui.hotkey.GlobalHotkeyListener")
    @patch("voicepad.tui.overlay.StatusOverlay")
    def test_registration_failure_does_not_mark_listener_active(
        self, mock_overlay_class: Mock, mock_listener_class: Mock, mock_app: Mock
    ) -> None:
        """A Windows registration error leaves the app listener state inactive."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_listener_class.return_value.start.side_effect = PermissionError("denied")
        handler = HotkeyHandler(mock_app)

        with patch("voicepad.tui.handlers.hotkey_handler.platform.system", return_value="Windows"):
            handler.start_hotkey_listener()

        assert mock_app._hotkey_listener is None

    @patch("voicepad.tui.hotkey.GlobalHotkeyListener")
    @patch("voicepad.tui.overlay.StatusOverlay")
    def test_handles_exception_gracefully(
        self, mock_overlay_class: Mock, mock_listener_class: Mock, mock_app: Mock
    ) -> None:
        """Test that start_hotkey_listener handles exceptions gracefully."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_overlay_class.side_effect = RuntimeError("Test error")

        handler = HotkeyHandler(mock_app)
        # Should not raise
        handler.start_hotkey_listener()


class TestHotkeyOnToggle:
    """Test hotkey_on_toggle and hotkey_toggle_recording methods."""

    def test_dispatches_toggle_to_textual_thread(self, mock_app: Mock) -> None:
        """A request from a listener thread is dispatched through Textual."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        handler = HotkeyHandler(mock_app)
        handler.hotkey_on_toggle()

        mock_app.call_from_thread.assert_called_once_with(handler.hotkey_toggle_recording)

    def test_toggle_starts_when_idle(self, mock_app: Mock) -> None:
        """An idle app starts recording when toggled."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "hotkey_start_recording") as start_recording:
            handler.hotkey_toggle_recording()

        start_recording.assert_called_once_with()

    def test_toggle_stops_when_recording(self, mock_app: Mock) -> None:
        """An active recording stops when toggled."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._recording = True
        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "hotkey_stop_recording") as stop_recording:
            handler.hotkey_toggle_recording()

        stop_recording.assert_called_once_with()


class TestHotkeyStartRecording:
    """Test hotkey_start_recording method."""

    def test_returns_early_when_already_recording(self, mock_app: Mock) -> None:
        """Test that hotkey_start_recording returns early when already recording."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._recording = True
        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "overlay_set") as mock_overlay_set:
            handler.hotkey_start_recording()
            mock_overlay_set.assert_not_called()

    def test_returns_early_when_transcribing(self, mock_app: Mock) -> None:
        """Test that hotkey_start_recording returns early when transcribing."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._transcribing = True
        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "overlay_set") as mock_overlay_set:
            handler.hotkey_start_recording()
            mock_overlay_set.assert_not_called()

    def test_sets_error_overlay_when_model_not_ready(self, mock_app: Mock) -> None:
        """Test that hotkey_start_recording sets error overlay when model not ready."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._model_ready = False
        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "overlay_set") as mock_overlay_set:
            handler.hotkey_start_recording()
            mock_overlay_set.assert_called_once_with("error")

    def test_switches_to_record_tab(self, mock_app: Mock) -> None:
        """Test that hotkey_start_recording switches to record tab."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_tabs = Mock()
        mock_app.query_one.return_value = mock_tabs

        handler = HotkeyHandler(mock_app)
        handler.hotkey_start_recording()

        mock_app.query_one.assert_called_once_with("#tabs", ANY)
        assert mock_tabs.active == "tab-record"

    def test_sets_recording_overlay(self, mock_app: Mock) -> None:
        """Test that hotkey_start_recording sets recording overlay."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "overlay_set") as mock_overlay_set:
            handler.hotkey_start_recording()
            mock_overlay_set.assert_called_once_with("recording")

    def test_uses_tui_recording_handler(self, mock_app: Mock) -> None:
        """A desktop start request uses the TUI's shared recording handler."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        handler = HotkeyHandler(mock_app)
        handler.hotkey_start_recording()

        mock_app._recording_handler.start_recording.assert_called_once_with()

    def test_handles_tab_switch_exception(self, mock_app: Mock) -> None:
        """Test that hotkey_start_recording handles tab switch exceptions."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app.query_one.side_effect = RuntimeError("Test error")

        handler = HotkeyHandler(mock_app)
        # Should not raise
        handler.hotkey_start_recording()


class TestHotkeyStopRecording:
    """Test hotkey_stop_recording method."""

    def test_returns_early_when_not_recording(self, mock_app: Mock) -> None:
        """Test that hotkey_stop_recording returns early when not recording."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._recording = False
        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "overlay_set") as mock_overlay_set:
            handler.hotkey_stop_recording()
            mock_overlay_set.assert_not_called()

    def test_sets_hotkey_pending_copy_flag(self, mock_app: Mock) -> None:
        """Test that hotkey_stop_recording sets hotkey pending copy flag."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._recording = True
        handler = HotkeyHandler(mock_app)

        handler.hotkey_stop_recording()

        assert mock_app._hotkey_pending_copy is True

    def test_sets_transcribing_overlay(self, mock_app: Mock) -> None:
        """Test that hotkey_stop_recording sets transcribing overlay."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._recording = True
        handler = HotkeyHandler(mock_app)

        with patch.object(handler, "overlay_set") as mock_overlay_set:
            handler.hotkey_stop_recording()
            mock_overlay_set.assert_called_once_with("transcribing")

    def test_uses_tui_recording_handler(self, mock_app: Mock) -> None:
        """A desktop stop request uses the TUI's shared recording handler."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._recording = True

        handler = HotkeyHandler(mock_app)
        handler.hotkey_stop_recording()

        mock_app._recording_handler.stop_recording.assert_called_once_with()


class TestOverlaySet:
    """Test overlay_set method."""

    def test_does_nothing_when_overlay_is_none(self, mock_app: Mock) -> None:
        """Test that overlay_set does nothing when overlay is None."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_app._overlay = None
        handler = HotkeyHandler(mock_app)

        # Should not raise
        handler.overlay_set("recording")

    def test_calls_set_state_on_overlay(self, mock_app: Mock) -> None:
        """Test that overlay_set calls set_state on overlay."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_overlay = Mock()
        mock_app._overlay = mock_overlay
        handler = HotkeyHandler(mock_app)

        handler.overlay_set("recording")

        mock_overlay.set_state.assert_called_once_with("recording")

    def test_handles_exception_gracefully(self, mock_app: Mock) -> None:
        """Test that overlay_set handles exceptions gracefully."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_overlay = Mock()
        mock_overlay.set_state.side_effect = RuntimeError("Test error")
        mock_app._overlay = mock_overlay
        handler = HotkeyHandler(mock_app)

        # Should not raise
        handler.overlay_set("recording")

    def test_sets_different_states(self, mock_app: Mock) -> None:
        """Test that overlay_set can set different states."""
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler

        mock_overlay = Mock()
        mock_app._overlay = mock_overlay
        handler = HotkeyHandler(mock_app)

        states = ["recording", "transcribing", "copied", "error", "hidden"]
        for state in states:
            handler.overlay_set(state)

        assert mock_overlay.set_state.call_count == len(states)
        for i, state in enumerate(states):
            assert mock_overlay.set_state.call_args_list[i][0][0] == state
