"""Tests for RecordingTimer."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from voicepad.tui.utils.timer import RecordingTimer


class TestRecordingTimer:
    """Test suite for RecordingTimer class."""

    def test_init_stores_callback(self) -> None:
        """RecordingTimer stores the callback function."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        assert timer._on_tick == callback
        assert timer._running is False
        assert timer._thread is None

    def test_start_sets_running_flag(self) -> None:
        """start() sets the running flag to True."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()

        assert timer._running is True
        assert timer._thread is not None

        # Cleanup
        timer.stop()

    def test_start_creates_daemon_thread(self) -> None:
        """start() creates a daemon thread."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()

        assert timer._thread is not None
        assert timer._thread.daemon is True

        # Cleanup
        timer.stop()

    def test_start_when_already_running_does_nothing(self) -> None:
        """start() does nothing if timer is already running."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()
        first_thread = timer._thread

        timer.start()  # Second call

        assert timer._thread == first_thread

        # Cleanup
        timer.stop()

    def test_stop_clears_running_flag(self) -> None:
        """stop() clears the running flag."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()
        timer.stop()

        assert timer._running is False
        assert timer._thread is None

    def test_stop_when_not_running_does_nothing(self) -> None:
        """stop() does nothing if timer is not running."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        # Should not raise
        timer.stop()

        assert timer._running is False

    def test_timer_calls_callback_with_formatted_time(self) -> None:
        """Timer calls callback with formatted time strings."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()
        time.sleep(0.3)  # Wait for at least 2 ticks
        timer.stop()

        # Should have been called at least once
        assert callback.call_count >= 1

        # Check that callback was called with a string
        for call in callback.call_args_list:
            args = call[0]
            assert len(args) == 1
            assert isinstance(args[0], str)

    def test_timer_formats_seconds_for_short_duration(self) -> None:
        """Timer formats time as seconds for durations under 1 minute."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()
        time.sleep(0.3)
        timer.stop()

        # At least one call should have seconds format (e.g., "0.2s")
        called_with_seconds = any(
            "s" in str(call[0][0]) and ":" not in str(call[0][0]) for call in callback.call_args_list
        )
        assert called_with_seconds

    def test_timer_handles_callback_exceptions(self) -> None:
        """Timer continues running even if callback raises exception."""
        callback = MagicMock(side_effect=Exception("Callback error"))
        timer = RecordingTimer(on_tick=callback)

        timer.start()
        time.sleep(0.3)
        timer.stop()

        # Timer should have attempted multiple calls despite exceptions
        assert callback.call_count >= 1

    def test_timer_updates_at_regular_intervals(self) -> None:
        """Timer updates at approximately 0.1s intervals."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()
        time.sleep(0.5)  # Run for 0.5 seconds
        timer.stop()

        # Should have been called approximately 5 times (0.5s / 0.1s)
        # Allow some tolerance for timing variations
        assert 3 <= callback.call_count <= 7

    def test_multiple_start_stop_cycles(self) -> None:
        """Timer can be started and stopped multiple times."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        # First cycle
        timer.start()
        time.sleep(0.2)
        timer.stop()
        first_count = callback.call_count

        # Reset mock
        callback.reset_mock()

        # Second cycle
        timer.start()
        time.sleep(0.2)
        timer.stop()
        second_count = callback.call_count

        # Both cycles should have resulted in callbacks
        assert first_count >= 1
        assert second_count >= 1

    def test_timer_resets_start_time_on_each_start(self) -> None:
        """Timer resets start time each time start() is called."""
        callback = MagicMock()
        timer = RecordingTimer(on_tick=callback)

        timer.start()
        first_start = timer._start_time
        timer.stop()

        time.sleep(0.1)

        timer.start()
        second_start = timer._start_time
        timer.stop()

        # Start times should be different
        assert second_start > first_start
