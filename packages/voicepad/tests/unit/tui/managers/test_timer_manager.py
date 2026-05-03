"""Tests for TimerManager."""

from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_app() -> Mock:
    """Create a mock VoicePadApp instance."""
    app = Mock()
    app._recording = False
    app._timer_thread = None
    app._record_start = 0.0
    return app


class TestTimerManagerInit:
    """Test TimerManager initialization."""

    def test_init_stores_app_reference(self, mock_app: Mock) -> None:
        """Test that __init__ stores the app reference."""
        from voicepad.tui.managers.timer_manager import TimerManager

        manager = TimerManager(mock_app)
        assert manager.app is mock_app


class TestStartTimer:
    """Test start_timer method."""

    def test_creates_daemon_thread(self, mock_app: Mock) -> None:
        """Test that start_timer creates a daemon thread."""
        from voicepad.tui.managers.timer_manager import TimerManager

        manager = TimerManager(mock_app)

        with patch.object(manager, "_timer_loop"):
            manager.start_timer()

        assert mock_app._timer_thread is not None
        assert mock_app._timer_thread.daemon is True

    def test_starts_thread(self, mock_app: Mock) -> None:
        """Test that start_timer starts the thread."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_app._recording = True
        mock_app._record_start = time.monotonic()
        manager = TimerManager(mock_app)

        manager.start_timer()
        # Give thread time to start
        time.sleep(0.05)

        assert mock_app._timer_thread.is_alive()

        # Clean up
        mock_app._recording = False
        mock_app._timer_thread.join(timeout=1)


class TestStopTimer:
    """Test stop_timer method."""

    def test_clears_timer_thread(self, mock_app: Mock) -> None:
        """Test that stop_timer clears the timer thread."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_app._timer_thread = Mock()
        manager = TimerManager(mock_app)

        manager.stop_timer()

        assert mock_app._timer_thread is None

    def test_calls_refresh_status_label(self, mock_app: Mock) -> None:
        """Test that stop_timer calls refresh_status_label."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_app._timer_thread = Mock()
        manager = TimerManager(mock_app)

        with patch.object(manager, "_refresh_status_label") as mock_refresh:
            manager.stop_timer()
            mock_app.call_from_thread.assert_called_once_with(mock_refresh)

    def test_handles_exception_gracefully(self, mock_app: Mock) -> None:
        """Test that stop_timer handles exceptions gracefully."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_app._timer_thread = Mock()
        mock_app.call_from_thread.side_effect = RuntimeError("Test error")
        manager = TimerManager(mock_app)

        # Should not raise
        manager.stop_timer()
        assert mock_app._timer_thread is None


class TestTimerLoop:
    """Test _timer_loop method."""

    def test_updates_status_while_recording(self, mock_app: Mock) -> None:
        """Test that _timer_loop updates status while recording."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_app._recording = True
        mock_app._record_start = time.monotonic()
        mock_app.call_from_thread = Mock()
        manager = TimerManager(mock_app)

        # Run loop for a short time
        import threading

        thread = threading.Thread(target=manager._timer_loop, daemon=True)
        thread.start()
        time.sleep(0.25)  # Let it run for at least 2 iterations
        mock_app._recording = False  # Stop the loop
        thread.join(timeout=1)

        # Should have been called at least once
        assert mock_app.call_from_thread.call_count >= 1
        # First arg should be the update method
        assert mock_app.call_from_thread.call_args_list[0][0][0] == manager._update_status_with_timer

    def test_formats_time_with_minutes(self, mock_app: Mock) -> None:
        """Test that _timer_loop formats time with minutes for long recordings."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_app._recording = True
        mock_app._record_start = time.monotonic() - 65  # 65 seconds ago
        mock_app.call_from_thread = Mock()
        manager = TimerManager(mock_app)

        import threading

        thread = threading.Thread(target=manager._timer_loop, daemon=True)
        thread.start()
        time.sleep(0.25)
        mock_app._recording = False
        thread.join(timeout=1)

        # Check that at least one call used MM:SS format
        # call_from_thread is called with (method, timer_str)
        calls = [str(call[0][1]) for call in mock_app.call_from_thread.call_args_list]
        assert any(":" in call for call in calls)

    def test_formats_time_with_seconds_for_short_duration(self, mock_app: Mock) -> None:
        """Test that _timer_loop formats time with seconds for short recordings."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_app._recording = True
        mock_app._record_start = time.monotonic() - 5  # 5 seconds ago
        mock_app.call_from_thread = Mock()
        manager = TimerManager(mock_app)

        import threading

        thread = threading.Thread(target=manager._timer_loop, daemon=True)
        thread.start()
        time.sleep(0.25)
        mock_app._recording = False
        thread.join(timeout=1)

        # Check that at least one call used decimal seconds format
        # call_from_thread is called with (method, timer_str)
        calls = [str(call[0][1]) for call in mock_app.call_from_thread.call_args_list]
        assert any("s" in call and ":" not in call for call in calls)


class TestUpdateStatusWithTimer:
    """Test _update_status_with_timer method."""

    def test_updates_label_with_timer_string(self, mock_app: Mock) -> None:
        """Test that _update_status_with_timer updates the label."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_label = Mock()
        mock_app.query_one.return_value = mock_label
        manager = TimerManager(mock_app)

        manager._update_status_with_timer("01:23")

        mock_label.update.assert_called_once()
        call_arg = mock_label.update.call_args[0][0]
        assert "01:23" in call_arg
        assert "\U000f044a" in call_arg  # recording icon
        assert "\U000f051b" in call_arg  # timer icon


class TestRefreshStatusLabel:
    """Test _refresh_status_label method."""

    def test_updates_to_transcribing_when_timer_present(self, mock_app: Mock) -> None:
        """Test that _refresh_status_label updates to transcribing when timer is present."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_label = Mock()
        mock_label.render.return_value = "\U000f051b 01:23"
        mock_app.query_one.return_value = mock_label
        manager = TimerManager(mock_app)

        manager._refresh_status_label()

        mock_label.update.assert_called_once()
        call_arg = mock_label.update.call_args[0][0]
        assert "transcribing" in call_arg

    def test_does_not_update_when_no_timer(self, mock_app: Mock) -> None:
        """Test that _refresh_status_label doesn't update when no timer is present."""
        from voicepad.tui.managers.timer_manager import TimerManager

        mock_label = Mock()
        mock_label.render.return_value = "ready"
        mock_app.query_one.return_value = mock_label
        manager = TimerManager(mock_app)

        manager._refresh_status_label()

        mock_label.update.assert_not_called()
