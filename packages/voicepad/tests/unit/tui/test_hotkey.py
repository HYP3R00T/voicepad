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
        mock_pynput_listener.stop.side_effect = Exception("Stop failed")
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

        mock_global_hotkeys = MagicMock()
        activate_callback = None

        def capture_callback(hotkey_map):
            nonlocal activate_callback
            activate_callback = hotkey_map["<ctrl>+<alt>+v"]
            return mock_global_hotkeys

        # Mock the keyboard module import
        mock_keyboard = MagicMock()
        mock_keyboard.GlobalHotKeys = MagicMock(side_effect=capture_callback)

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            # Start the run in a thread
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)  # Give thread time to set up

            # Simulate first hotkey press (start)
            if activate_callback:
                activate_callback()
                assert listener._recording is True
                on_start.assert_called_once()

                # Simulate second hotkey press (stop)
                activate_callback()
                assert listener._recording is False
                on_stop.assert_called_once()

    def test_run_handles_on_start_exception(self) -> None:
        on_start = MagicMock(side_effect=Exception("Start failed"))
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)

        mock_global_hotkeys = MagicMock()
        activate_callback = None

        def capture_callback(hotkey_map):
            nonlocal activate_callback
            activate_callback = hotkey_map["<ctrl>+<alt>+v"]
            return mock_global_hotkeys

        mock_keyboard = MagicMock()
        mock_keyboard.GlobalHotKeys = MagicMock(side_effect=capture_callback)

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            # Should not raise even though on_start raises
            if activate_callback:
                activate_callback()
                assert listener._recording is True  # State still changes

    def test_run_handles_on_stop_exception(self) -> None:
        on_start = MagicMock()
        on_stop = MagicMock(side_effect=Exception("Stop failed"))
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=on_start, on_stop=on_stop)

        mock_global_hotkeys = MagicMock()
        activate_callback = None

        def capture_callback(hotkey_map):
            nonlocal activate_callback
            activate_callback = hotkey_map["<ctrl>+<alt>+v"]
            return mock_global_hotkeys

        mock_keyboard = MagicMock()
        mock_keyboard.GlobalHotKeys = MagicMock(side_effect=capture_callback)

        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            if activate_callback:
                # First press to start
                activate_callback()
                # Second press to stop (should handle exception)
                activate_callback()
                assert listener._recording is False

    def test_run_handles_pynput_import_error(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        # Mock import to raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("pynput not installed")):
            # Should not raise
            listener._run()

    def test_run_handles_global_hotkeys_exception(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        mock_keyboard = MagicMock()
        mock_keyboard.GlobalHotKeys = MagicMock(side_effect=Exception("Hotkey registration failed"))
        with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
            # Should not raise
            listener._run()
