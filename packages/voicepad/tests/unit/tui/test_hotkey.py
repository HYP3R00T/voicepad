"""Tests for GlobalHotkeyListener."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from voicepad.tui.hotkey import GlobalHotkeyListener, _parse_hotkey


class TestParseHotkey:
    """Test suite for _parse_hotkey function."""

    def test_parse_hotkey_returns_non_empty_string(self) -> None:
        """_parse_hotkey returns the string if non-empty."""
        result = _parse_hotkey("<ctrl>+<alt>+v")
        assert result == "<ctrl>+<alt>+v"

    def test_parse_hotkey_strips_whitespace(self) -> None:
        """_parse_hotkey strips leading and trailing whitespace."""
        result = _parse_hotkey("  <ctrl>+v  ")
        assert result == "<ctrl>+v"

    def test_parse_hotkey_returns_none_for_empty_string(self) -> None:
        """_parse_hotkey returns None for empty string."""
        result = _parse_hotkey("")
        assert result is None

    def test_parse_hotkey_returns_none_for_whitespace_only(self) -> None:
        """_parse_hotkey returns None for whitespace-only string."""
        result = _parse_hotkey("   ")
        assert result is None


class TestGlobalHotkeyListener:
    """Test suite for GlobalHotkeyListener class."""

    def test_init_stores_callbacks(self) -> None:
        """GlobalHotkeyListener stores hotkey and callbacks."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+<alt>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        assert listener._hotkey_str == "<ctrl>+<alt>+v"
        assert listener._on_start == on_start
        assert listener._on_stop == on_stop
        assert listener._recording is False
        assert listener._listener is None
        assert listener._thread is None

    @patch("voicepad.tui.hotkey.logger")
    def test_start_with_empty_hotkey_does_nothing(self, mock_logger: MagicMock) -> None:
        """start() does nothing if hotkey is empty."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="",
            on_start=on_start,
            on_stop=on_stop,
        )

        listener.start()

        assert listener._thread is None
        mock_logger.info.assert_called_once()
        assert "disabled" in str(mock_logger.info.call_args).lower()

    @patch("voicepad.tui.hotkey.logger")
    def test_start_creates_daemon_thread(self, mock_logger: MagicMock) -> None:
        """start() creates a daemon thread."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        with patch.object(listener, "_run"):
            listener.start()

            assert listener._thread is not None
            assert listener._thread.daemon is True
            assert listener._thread.name == "hotkey-listener"
            mock_logger.info.assert_called()

    @patch("voicepad.tui.hotkey.logger")
    def test_stop_stops_listener(self, mock_logger: MagicMock) -> None:
        """stop() stops the pynput listener."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        mock_pynput_listener = MagicMock()
        listener._listener = mock_pynput_listener

        listener.stop()

        mock_pynput_listener.stop.assert_called_once()
        mock_logger.info.assert_called_once()
        assert "stopped" in str(mock_logger.info.call_args).lower()

    @patch("voicepad.tui.hotkey.logger")
    def test_stop_handles_exception_gracefully(self, mock_logger: MagicMock) -> None:
        """stop() handles exceptions when stopping listener."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        mock_pynput_listener = MagicMock()
        mock_pynput_listener.stop.side_effect = Exception("Stop failed")
        listener._listener = mock_pynput_listener

        # Should not raise
        listener.stop()

        mock_logger.info.assert_called_once()

    @patch("voicepad.tui.hotkey.logger")
    def test_stop_when_listener_is_none(self, mock_logger: MagicMock) -> None:
        """stop() works when listener is None."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        # Should not raise
        listener.stop()

        mock_logger.info.assert_called_once()

    @patch("pynput.keyboard")
    @patch("voicepad.tui.hotkey.logger")
    def test_run_creates_global_hotkeys(self, mock_logger: MagicMock, mock_keyboard: MagicMock) -> None:
        """_run() creates GlobalHotKeys with correct mapping."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        mock_global_hotkeys = MagicMock()
        mock_keyboard.GlobalHotKeys.return_value = mock_global_hotkeys

        listener._run()

        # Verify GlobalHotKeys was created with a mapping
        mock_keyboard.GlobalHotKeys.assert_called_once()
        hotkey_map = mock_keyboard.GlobalHotKeys.call_args[0][0]
        assert "<ctrl>+v" in hotkey_map
        assert callable(hotkey_map["<ctrl>+v"])

        # Verify run was called
        mock_global_hotkeys.run.assert_called_once()

    @patch("pynput.keyboard")
    def test_run_with_empty_hotkey_returns_early(self, mock_keyboard: MagicMock) -> None:
        """_run() returns early if hotkey is empty."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="",
            on_start=on_start,
            on_stop=on_stop,
        )

        listener._run()

        mock_keyboard.GlobalHotKeys.assert_not_called()

    @patch("pynput.keyboard")
    @patch("voicepad.tui.hotkey.logger")
    def test_run_handles_import_error(self, mock_logger: MagicMock, mock_keyboard: MagicMock) -> None:
        """_run() handles pynput import errors."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        mock_keyboard.GlobalHotKeys.side_effect = ImportError("pynput not installed")

        listener._run()

        mock_logger.error.assert_called_once()
        assert "failed" in str(mock_logger.error.call_args).lower()

    @patch("pynput.keyboard")
    def test_hotkey_activation_toggles_recording(self, mock_keyboard: MagicMock) -> None:
        """Hotkey activation toggles recording state."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        # Capture the activation callback
        activation_callback = None

        def capture_callback(hotkey_map):
            nonlocal activation_callback
            activation_callback = hotkey_map["<ctrl>+v"]
            mock_listener = MagicMock()
            mock_listener.run = MagicMock()  # Don't actually run
            return mock_listener

        mock_keyboard.GlobalHotKeys.side_effect = capture_callback

        listener._run()

        assert activation_callback is not None
        assert callable(activation_callback)

        # First press: start recording
        activation_callback()
        assert listener._recording is True
        on_start.assert_called_once()
        on_stop.assert_not_called()

        # Second press: stop recording
        activation_callback()
        assert listener._recording is False
        on_start.assert_called_once()  # Still only once
        on_stop.assert_called_once()

    @patch("pynput.keyboard")
    @patch("voicepad.tui.hotkey.logger")
    def test_hotkey_activation_handles_on_start_error(self, mock_logger: MagicMock, mock_keyboard: MagicMock) -> None:
        """Hotkey activation handles errors in on_start callback."""
        on_start = MagicMock(side_effect=Exception("Start failed"))
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        # Capture the activation callback
        activation_callback = None

        def capture_callback(hotkey_map):
            nonlocal activation_callback
            activation_callback = hotkey_map["<ctrl>+v"]
            mock_listener = MagicMock()
            mock_listener.run = MagicMock()
            return mock_listener

        mock_keyboard.GlobalHotKeys.side_effect = capture_callback

        listener._run()

        assert activation_callback is not None
        # Should not raise
        activation_callback()

        mock_logger.error.assert_called()
        assert "on_start" in str(mock_logger.error.call_args).lower()

    @patch("pynput.keyboard")
    @patch("voicepad.tui.hotkey.logger")
    def test_hotkey_activation_handles_on_stop_error(self, mock_logger: MagicMock, mock_keyboard: MagicMock) -> None:
        """Hotkey activation handles errors in on_stop callback."""
        on_start = MagicMock()
        on_stop = MagicMock(side_effect=Exception("Stop failed"))

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        # Capture the activation callback
        activation_callback = None

        def capture_callback(hotkey_map):
            nonlocal activation_callback
            activation_callback = hotkey_map["<ctrl>+v"]
            mock_listener = MagicMock()
            mock_listener.run = MagicMock()
            return mock_listener

        mock_keyboard.GlobalHotKeys.side_effect = capture_callback

        listener._run()

        assert activation_callback is not None
        # Start recording first
        activation_callback()
        assert listener._recording is True

        # Stop recording - should handle error
        activation_callback()

        mock_logger.error.assert_called()
        assert "on_stop" in str(mock_logger.error.call_args).lower()

    @patch("pynput.keyboard")
    @patch("voicepad.tui.hotkey.logger")
    def test_multiple_activations_toggle_correctly(self, mock_logger: MagicMock, mock_keyboard: MagicMock) -> None:
        """Multiple hotkey activations toggle recording correctly."""
        on_start = MagicMock()
        on_stop = MagicMock()

        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+v",
            on_start=on_start,
            on_stop=on_stop,
        )

        # Capture the activation callback
        activation_callback = None

        def capture_callback(hotkey_map):
            nonlocal activation_callback
            activation_callback = hotkey_map["<ctrl>+v"]
            mock_listener = MagicMock()
            mock_listener.run = MagicMock()
            return mock_listener

        mock_keyboard.GlobalHotKeys.side_effect = capture_callback

        listener._run()

        assert activation_callback is not None
        # Multiple toggles
        activation_callback()  # Start
        assert listener._recording is True

        activation_callback()  # Stop
        assert listener._recording is False

        activation_callback()  # Start again
        assert listener._recording is True

        activation_callback()  # Stop again
        assert listener._recording is False

        assert on_start.call_count == 2
        assert on_stop.call_count == 2
