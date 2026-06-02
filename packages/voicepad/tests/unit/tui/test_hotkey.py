"""Tests for voicepad.tui.hotkey."""

from __future__ import annotations

import threading
import time
from typing import cast
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

    def test_stop_calls_remove_hotkey(self) -> None:
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_start=MagicMock(), on_stop=MagicMock())
        mock_keyboard = MagicMock()
        with patch.dict("sys.modules", {"keyboard": mock_keyboard}):
            listener.stop()
            # Should attempt to remove the hotkey
            mock_keyboard.remove_hotkey.assert_called_once_with("ctrl+alt+v")

    def test_stop_handles_exception_gracefully(self) -> None:
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_start=MagicMock(), on_stop=MagicMock())
        mock_keyboard = MagicMock()
        mock_keyboard.remove_hotkey.side_effect = RuntimeError("Remove failed")
        with patch.dict("sys.modules", {"keyboard": mock_keyboard}):
            # Should not raise
            listener.stop()

    def test_run_returns_early_with_empty_hotkey(self) -> None:
        listener = GlobalHotkeyListener(hotkey="", on_start=MagicMock(), on_stop=MagicMock())
        # Should return immediately without error
        listener._run()

    def test_run_toggles_recording_state(self) -> None:
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_start=on_start, on_stop=on_stop)

        # Mock keyboard module
        mock_keyboard = MagicMock()
        hotkey_callback = None

        def capture_callback(hotkey, callback, suppress=False):
            nonlocal hotkey_callback
            hotkey_callback = callback

        mock_keyboard.add_hotkey = MagicMock(side_effect=capture_callback)
        mock_keyboard.wait = MagicMock()

        with patch.dict("sys.modules", {"keyboard": mock_keyboard}):
            # Start the run in a thread
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)  # Give thread time to set up

            # Simulate hotkey press (should trigger start)
            if hotkey_callback:
                hotkey_callback()
                time.sleep(0.05)
                assert listener._recording is True
                on_start.assert_called_once()

                # Simulate second press (should trigger stop)
                hotkey_callback()
                time.sleep(0.05)
                assert listener._recording is False
                on_stop.assert_called_once()

    def test_run_handles_on_start_exception(self) -> None:
        on_start = MagicMock(side_effect=Exception("Start failed"))
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_start=on_start, on_stop=on_stop)

        mock_keyboard = MagicMock()
        hotkey_callback = None

        def capture_callback(hotkey, callback, suppress=False):
            nonlocal hotkey_callback
            hotkey_callback = callback

        mock_keyboard.add_hotkey = MagicMock(side_effect=capture_callback)
        mock_keyboard.wait = MagicMock()

        with patch.dict("sys.modules", {"keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            # Should not raise even though on_start raises
            if hotkey_callback:
                hotkey_callback()
                time.sleep(0.05)
                assert listener._recording is True  # State still changes

    def test_run_handles_on_stop_exception(self) -> None:
        on_start = MagicMock()
        on_stop = MagicMock(side_effect=Exception("Stop failed"))
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_start=on_start, on_stop=on_stop)

        mock_keyboard = MagicMock()
        hotkey_callback = None

        def capture_callback(hotkey, callback, suppress=False):
            nonlocal hotkey_callback
            hotkey_callback = callback

        mock_keyboard.add_hotkey = MagicMock(side_effect=capture_callback)
        mock_keyboard.wait = MagicMock()

        with patch.dict("sys.modules", {"keyboard": mock_keyboard}):
            thread = threading.Thread(target=listener._run, daemon=True)
            thread.start()
            time.sleep(0.1)

            if hotkey_callback:
                # First press to start
                hotkey_callback()
                time.sleep(0.05)
                # Second press to stop (should handle exception)
                hotkey_callback()
                time.sleep(0.05)
                assert listener._recording is False

    def test_run_handles_keyboard_import_error(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        # Mock import to raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("keyboard not installed")):
            # Should not raise
            listener._run()

    def test_run_handles_listener_exception(self) -> None:
        listener = GlobalHotkeyListener(hotkey="<ctrl>+<alt>+v", on_start=MagicMock(), on_stop=MagicMock())
        mock_keyboard = MagicMock()
        mock_keyboard.add_hotkey = MagicMock(side_effect=Exception("Hotkey registration failed"))
        with patch.dict("sys.modules", {"keyboard": mock_keyboard}):
            # Should not raise
            listener._run()

    def test_hotkey_callback_uses_lock(self) -> None:
        """Test that the hotkey callback uses thread lock for safety."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_start=on_start, on_stop=on_stop)

        # Replace the lock with a mock to verify it's being used
        mock_lock = MagicMock()
        listener._lock = cast(threading.Lock, mock_lock)

        # Call the hotkey callback
        listener._on_hotkey()

        # Verify the lock's __enter__ and __exit__ were called (context manager usage)
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
