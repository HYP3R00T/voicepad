"""Recording timer management for VoicePad TUI."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp


class TimerManager:
    """Manages the recording timer display."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app

    def start_timer(self) -> None:
        """Start the recording timer thread."""
        self.app._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.app._timer_thread.start()

    def stop_timer(self) -> None:
        """Stop the recording timer thread."""
        self.app._timer_thread = None
        with contextlib.suppress(Exception):
            self.app.call_from_thread(self._refresh_status_label)

    def _timer_loop(self) -> None:
        """Timer loop that updates the status label with elapsed time."""
        while self.app._recording:
            elapsed = time.monotonic() - self.app._record_start
            mins, secs = divmod(int(elapsed), 60)
            timer_str = f"{mins:02d}:{secs:02d}" if mins else f"{elapsed:.1f}s"
            with contextlib.suppress(Exception):
                self.app.call_from_thread(self._update_status_with_timer, timer_str)
            time.sleep(0.1)

    def _update_status_with_timer(self, timer_str: str) -> None:
        """Update the status label with the current timer value."""
        from textual.widgets import Label

        label = self.app.query_one("#status", Label)
        label.update(f"\U000f044a  recording…  \U000f051b {timer_str}")

    def _refresh_status_label(self) -> None:
        """Refresh the status label after timer stops."""
        from textual.widgets import Label

        label = self.app.query_one("#status", Label)
        if "\U000f051b" in str(label.render()):
            label.update("\U000f051f  transcribing…")
