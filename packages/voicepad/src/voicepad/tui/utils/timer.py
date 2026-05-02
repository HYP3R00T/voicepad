"""Recording timer utilities for VoicePad TUI."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable


class RecordingTimer:
    """A background timer for tracking recording duration.

    Runs in a separate thread and calls a callback function with
    formatted time strings at regular intervals.
    """

    def __init__(self, on_tick: Callable[[str], None]) -> None:
        """Initialize the recording timer.

        Args:
            on_tick: Callback function that receives formatted time strings.
        """
        self._on_tick = on_tick
        self._thread: threading.Thread | None = None
        self._running = False
        self._start_time: float = 0.0

    def start(self) -> None:
        """Start the timer in a background thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the timer."""
        self._running = False
        self._thread = None

    def _loop(self) -> None:
        """Background loop that updates the timer display."""
        while self._running:
            elapsed = time.monotonic() - self._start_time
            mins, secs = divmod(int(elapsed), 60)
            timer_str = f"{mins:02d}:{secs:02d}" if mins else f"{elapsed:.1f}s"
            with contextlib.suppress(Exception):
                self._on_tick(timer_str)
            time.sleep(0.1)
