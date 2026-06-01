"""Tests for voicepad.tui.hotkey."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from voicepad.tui.hotkey import GlobalHotkeyListener, _parse_hotkey

# ---------------------------------------------------------------------------
# _parse_hotkey
# ---------------------------------------------------------------------------


class TestParseHotkey:
    def test_returns_stripped_string(self) -> None:
        assert _parse_hotkey("  <ctrl>+<alt>+v  ") == "<ctrl>+<alt>+v"

    def test_returns_none_for_empty_string(self) -> None:
        assert _parse_hotkey("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        assert _parse_hotkey("   ") is None

    def test_returns_valid_hotkey(self) -> None:
        assert _parse_hotkey("<ctrl>+<shift>+r") == "<ctrl>+<shift>+r"


# ---------------------------------------------------------------------------
# GlobalHotkeyListener
# ---------------------------------------------------------------------------


class TestGlobalHotkeyListener:
    def test_init_stores_callbacks(self) -> None:
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)
        assert listener._on_start is on_start
        assert listener._on_stop is on_stop
        assert listener._hotkey_str == "<ctrl>+<alt>+v"

    def test_start_does_nothing_with_empty_hotkey(self) -> None:
        listener = GlobalHotkeyListener(hotkey="", on_start=MagicMock(), on_stop=MagicMock())
        listener.start()
        assert listener._thread is None

    def test_start_creates_thread_with_valid_hotkey(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        with patch.object(listener, "_run"):
            listener.start()
        assert listener._thread is not None
        assert listener._thread.daemon is True

    def test_stop_calls_listener_stop(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        mock_pynput_listener = MagicMock()
        listener._listener = mock_pynput_listener
        listener.stop()
        mock_pynput_listener.stop.assert_called_once()

    def test_stop_handles_exception_gracefully(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        mock_pynput_listener = MagicMock()
        mock_pynput_listener.stop.side_effect = RuntimeError("Stop failed")
        listener._listener = mock_pynput_listener
        # Should not raise
        listener.stop()

    def test_run_returns_early_with_empty_hotkey(self) -> None:
        listener = GlobalHotkeyListener(hotkey="", on_start=MagicMock(), on_stop=MagicMock())
        # Should return immediately without error
        listener._run()

    def test_run_toggles_recording_state(self) -> None:
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)

        mock_listener_instance = MagicMock()
        on_press_callback = None
        on_release_callback = None

        def capture_callbacks(on_press=None, on_release=None):
            nonlocal on_press_callback, on_release_callback
            on_press_callback = on_press
            on_release_callback = on_release
            return mock_listener_instance

        # Mock the keyboard module import
        mock_keyboard = MagicMock()
        mock_keyboard.Listener = MagicMock(side_effect=capture_callbacks)
        mock_keyboard.Key = MagicMock()
        mock_keyboard.Key.ctrl_l = "ctrl_l"
        mock_keyboard.Key.ctrl_r = "ctrl_r"
        mock_keyboard.Key.alt_l = "alt_l"
        mock_keyboard.Key.alt_r = "alt_r"
        mock_keyboard.KeyCode = MagicMock()
        mock_keyboard.KeyCode.from_char = MagicMock(return_value="v")

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            # Start the run in a thread
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)  # Give thread time to set up

            # Simulate pressing ctrl and alt
            if on_press_callback:
                on_press_callback("ctrl_l")
                on_press_callback("alt_l")
                # Simulate pressing v (should trigger start)
                on_press_callback("v")
                time.sleep(0.05)
                assert listener._recording is True
                on_start.assert_called_once()

                # Simulate releasing and pressing again (should trigger stop)
                on_release_callback("v")
                on_press_callback("v")
                time.sleep(0.05)
                assert listener._recording is False
                on_stop.assert_called_once()

    def test_run_handles_on_start_exception(self) -> None:
        on_start = MagicMock(side_effect=Exception("Start failed"))
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)

        mock_listener_instance = MagicMock()
        on_press_callback = None

        def capture_callbacks(on_press=None, on_release=None):
            nonlocal on_press_callback
            on_press_callback = on_press
            return mock_listener_instance

        mock_keyboard = MagicMock()
        mock_keyboard.Listener = MagicMock(side_effect=capture_callbacks)
        mock_keyboard.Key = MagicMock()
        mock_keyboard.Key.ctrl_l = "ctrl_l"
        mock_keyboard.Key.ctrl_r = "ctrl_r"
        mock_keyboard.Key.alt_l = "alt_l"
        mock_keyboard.Key.alt_r = "alt_r"
        mock_keyboard.KeyCode = MagicMock()
        mock_keyboard.KeyCode.from_char = MagicMock(return_value="v")

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            # Should not raise even though on_start raises
            if on_press_callback:
                on_press_callback("ctrl_l")
                on_press_callback("alt_l")
                on_press_callback("v")
                time.sleep(0.05)
                assert listener._recording is True  # State still changes

    def test_run_handles_on_stop_exception(self) -> None:
        on_start = MagicMock()
        on_stop = MagicMock(side_effect=Exception("Stop failed"))
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)

        mock_listener_instance = MagicMock()
        on_press_callback = None
        on_release_callback = None

        def capture_callbacks(on_press=None, on_release=None):
            nonlocal on_press_callback, on_release_callback
            on_press_callback = on_press
            on_release_callback = on_release
            return mock_listener_instance

        mock_keyboard = MagicMock()
        mock_keyboard.Listener = MagicMock(side_effect=capture_callbacks)
        mock_keyboard.Key = MagicMock()
        mock_keyboard.Key.ctrl_l = "ctrl_l"
        mock_keyboard.Key.ctrl_r = "ctrl_r"
        mock_keyboard.Key.alt_l = "alt_l"
        mock_keyboard.Key.alt_r = "alt_r"
        mock_keyboard.KeyCode = MagicMock()
        mock_keyboard.KeyCode.from_char = MagicMock(return_value="v")

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            if on_press_callback:
                # First press to start
                on_press_callback("ctrl_l")
                on_press_callback("alt_l")
                on_press_callback("v")
                time.sleep(0.05)
                # Second press to stop (should handle exception)
                on_release_callback("v")
                on_press_callback("v")
                time.sleep(0.05)
                assert listener._recording is False

    def test_run_handles_pynput_import_error(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        # Mock import to raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("pynput not installed")):
            # Should not raise
            listener._run()

    def test_run_handles_listener_exception(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        mock_keyboard = MagicMock()
        mock_keyboard.Listener = MagicMock(side_effect=Exception("Listener registration failed"))
        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            # Should not raise
            listener._run()

    def test_timeout_prevents_delayed_trigger(self) -> None:
        """Test that hotkey doesn't trigger if key is pressed too long after modifier."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)

        mock_listener_instance = MagicMock()
        on_press_callback = None

        def capture_callbacks(on_press=None, on_release=None):
            nonlocal on_press_callback
            on_press_callback = on_press
            return mock_listener_instance

        mock_keyboard = MagicMock()
        mock_keyboard.Listener = MagicMock(side_effect=capture_callbacks)
        mock_keyboard.Key = MagicMock()
        mock_keyboard.Key.ctrl_l = "ctrl_l"
        mock_keyboard.Key.ctrl_r = "ctrl_r"
        mock_keyboard.Key.alt_l = "alt_l"
        mock_keyboard.Key.alt_r = "alt_r"
        mock_keyboard.KeyCode = MagicMock()
        mock_keyboard.KeyCode.from_char = MagicMock(return_value="v")

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            if on_press_callback:
                # Press modifiers
                on_press_callback("ctrl_l")
                on_press_callback("alt_l")
                # Wait longer than timeout (0.5s)
                time.sleep(0.6)
                # Press key - should NOT trigger because of timeout
                on_press_callback("v")
                time.sleep(0.05)
                # Recording should not have started
                assert listener._recording is False
                on_start.assert_not_called()

    def test_simultaneous_press_triggers_hotkey(self) -> None:
        """Test that hotkey triggers when modifiers and key are pressed together."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)

        mock_listener_instance = MagicMock()
        on_press_callback = None

        def capture_callbacks(on_press=None, on_release=None):
            nonlocal on_press_callback
            on_press_callback = on_press
            return mock_listener_instance

        mock_keyboard = MagicMock()
        mock_keyboard.Listener = MagicMock(side_effect=capture_callbacks)
        mock_keyboard.Key = MagicMock()
        mock_keyboard.Key.ctrl_l = "ctrl_l"
        mock_keyboard.Key.ctrl_r = "ctrl_r"
        mock_keyboard.Key.alt_l = "alt_l"
        mock_keyboard.Key.alt_r = "alt_r"
        mock_keyboard.KeyCode = MagicMock()
        mock_keyboard.KeyCode.from_char = MagicMock(return_value="v")

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            if on_press_callback:
                # Press modifiers and key quickly (within timeout)
                on_press_callback("ctrl_l")
                on_press_callback("alt_l")
                time.sleep(0.05)  # Small delay, well within 0.5s timeout
                on_press_callback("v")
                time.sleep(0.05)
                # Recording should have started
                assert listener._recording is True
                on_start.assert_called_once()
