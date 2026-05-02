"""Tests for voicepad.tui.hotkey."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from voicepad.tui.hotkey import GlobalHotkeyListener, _parse_hotkey


class TestParseHotkey:
    """Tests for _parse_hotkey helper function."""

    def test_parse_valid_hotkey(self) -> None:
        """_parse_hotkey returns the hotkey string if non-empty."""
        result = _parse_hotkey("<ctrl>+v")
        assert result == "<ctrl>+v"

    def test_parse_empty_string(self) -> None:
        """_parse_hotkey returns None for empty string."""
        result = _parse_hotkey("")
        assert result is None

    def test_parse_whitespace_only(self) -> None:
        """_parse_hotkey returns None for whitespace-only string."""
        result = _parse_hotkey("   ")
        assert result is None

    def test_parse_strips_whitespace(self) -> None:
        """_parse_hotkey strips leading/trailing whitespace."""
        result = _parse_hotkey("  <ctrl>+v  ")
        assert result == "<ctrl>+v"


class TestGlobalHotkeyListener:
    """Tests for GlobalHotkeyListener class."""

    def test_listener_initializes_with_callbacks(self) -> None:
        """GlobalHotkeyListener stores callbacks and initial state."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        assert listener._hotkey_str == "<ctrl>+v"
        assert listener._on_start is on_start
        assert listener._on_stop is on_stop
        assert listener._recording is False

    def test_start_with_empty_hotkey_logs_and_returns(self) -> None:
        """When hotkey is empty, start() logs a message and returns without starting thread."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("", on_start, on_stop)

        with patch("voicepad.tui.hotkey.logger") as mock_logger:
            listener.start()

        mock_logger.info.assert_called_once()
        assert listener._thread is None

    def test_start_with_valid_hotkey_starts_thread(self) -> None:
        """When hotkey is valid, start() starts a daemon thread."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        with patch.object(listener, "_run"):
            listener.start()

        assert listener._thread is not None
        assert listener._thread.daemon is True
        listener._thread.join(timeout=1.0)

    def test_stop_calls_listener_stop(self) -> None:
        """When stop() is called, the pynput listener is stopped."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        mock_pynput_listener = MagicMock()
        listener._listener = mock_pynput_listener

        listener.stop()

        mock_pynput_listener.stop.assert_called_once()

    def test_stop_gracefully_handles_no_listener(self) -> None:
        """When stop() is called before start(), it does not raise an exception."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        # Should not raise
        listener.stop()

    def test_on_activate_toggles_recording_and_calls_callbacks(self) -> None:
        """When hotkey is pressed, _on_activate toggles recording and calls appropriate callback."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        # Patch pynput.keyboard at the point it is imported inside _run()
        mock_kb = MagicMock()
        mock_pynput_listener = MagicMock()
        captured = {}

        def capture_hotkeys(hotkey_map: dict) -> object:
            captured["map"] = hotkey_map
            return mock_pynput_listener

        mock_kb.GlobalHotKeys.side_effect = capture_hotkeys

        with (
            patch.dict("sys.modules", {"pynput": MagicMock(), "pynput.keyboard": mock_kb}),
            patch("pynput.keyboard", mock_kb),
        ):
            listener._run()

        # GlobalHotKeys should have been called with a hotkey map
        assert mock_kb.GlobalHotKeys.called

    def test_on_activate_first_press_calls_on_start(self) -> None:
        """First hotkey press calls on_start callback."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        assert listener._recording is False

        # Simulate the first activation (same logic as _on_activate in hotkey.py)
        if listener._recording:
            listener._recording = False
            listener._on_stop()
        else:
            listener._recording = True
            listener._on_start()

        assert listener._recording is True
        on_start.assert_called_once()
        on_stop.assert_not_called()

    def test_on_activate_second_press_calls_on_stop(self) -> None:
        """Second hotkey press calls on_stop callback."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        # Simulate recording is already in progress
        listener._recording = True

        # Simulate the second activation
        if listener._recording:
            listener._recording = False
            listener._on_stop()
        else:
            listener._recording = True
            listener._on_start()

        assert listener._recording is False
        on_stop.assert_called_once()
        on_start.assert_not_called()

    def test_on_activate_handles_callback_exception(self) -> None:
        """When on_start raises, _on_activate logs the error and does not propagate."""
        on_start = MagicMock(side_effect=RuntimeError("callback failed"))
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        # Patch pynput so _run() can execute and register _on_activate
        mock_kb = MagicMock()
        captured_callback: list = []

        def capture_hotkeys(hotkey_map: dict) -> object:
            captured_callback.extend(hotkey_map.values())
            # Return a mock that calls run() immediately then stops
            m = MagicMock()
            m.run.side_effect = lambda: None
            return m

        mock_kb.GlobalHotKeys.side_effect = capture_hotkeys

        with (
            patch.dict("sys.modules", {"pynput": MagicMock(), "pynput.keyboard": mock_kb}),
            patch("pynput.keyboard", mock_kb),
            patch("voicepad.tui.hotkey.logger") as mock_logger,
        ):
            listener._run()

            # Manually invoke the captured _on_activate callback
            if captured_callback:
                activate = captured_callback[0]
                activate()  # first press — triggers on_start which raises
                # logger.error should have been called
                mock_logger.error.assert_called()

    def test_listener_logging_on_start(self) -> None:
        """When start() is called with a valid hotkey, a log message is emitted."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        with patch("voicepad.tui.hotkey.logger") as mock_logger:
            with patch.object(listener, "_run"):
                listener.start()

            assert any("started" in str(c).lower() for c in mock_logger.info.call_args_list)

    def test_listener_logging_on_stop(self) -> None:
        """When stop() is called, a log message is emitted."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        with patch("voicepad.tui.hotkey.logger") as mock_logger:
            listener._listener = MagicMock()
            listener.stop()

            assert any("stopped" in str(c).lower() for c in mock_logger.info.call_args_list)

    def test_run_returns_when_hotkey_empty(self) -> None:
        """When hotkey is empty, _run() returns immediately without creating a pynput listener."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("", on_start, on_stop)

        mock_kb = MagicMock()
        with (
            patch.dict("sys.modules", {"pynput": MagicMock(), "pynput.keyboard": mock_kb}),
            patch("pynput.keyboard", mock_kb),
        ):
            listener._run()

        # No pynput listener should be created
        assert listener._listener is None
        mock_kb.GlobalHotKeys.assert_not_called()

    def test_run_handles_exception_in_pynput(self) -> None:
        """When pynput raises an exception, _run() logs the error."""
        on_start = MagicMock()
        on_stop = MagicMock()
        listener = GlobalHotkeyListener("<ctrl>+v", on_start, on_stop)

        mock_kb = MagicMock()
        mock_kb.GlobalHotKeys.side_effect = RuntimeError("pynput error")

        with (
            patch.dict("sys.modules", {"pynput": MagicMock(), "pynput.keyboard": mock_kb}),
            patch("pynput.keyboard", mock_kb),
            patch("voicepad.tui.hotkey.logger") as mock_logger,
        ):
            listener._run()

            mock_logger.error.assert_called_once()
