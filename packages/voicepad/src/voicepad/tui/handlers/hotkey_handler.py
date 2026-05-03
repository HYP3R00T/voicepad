"""Hotkey handler for VoicePad TUI."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from textual.widgets import TabbedContent

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp

logger = logging.getLogger(__name__)


class HotkeyHandler:
    """Handles global hotkey functionality."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app

    def start_hotkey_listener(self) -> None:
        """Start the system-wide hotkey listener and status overlay."""
        hotkey = getattr(self.app.config, "global_hotkey", "")
        if not hotkey:
            return
        try:
            from voicepad.tui.hotkey import GlobalHotkeyListener
            from voicepad.tui.overlay import StatusOverlay

            self.app._overlay = StatusOverlay()
            self.app._overlay.start()

            self.app._hotkey_listener = GlobalHotkeyListener(
                hotkey=hotkey,
                on_start=self.hotkey_on_start,
                on_stop=self.hotkey_on_stop,
            )
            self.app._hotkey_listener.start()
            logger.info(f"Global hotkey active: {hotkey}")
        except Exception as e:
            logger.warning(f"Could not start global hotkey listener: {e}")

    def hotkey_on_start(self) -> None:
        """Called from the hotkey thread when the hotkey is pressed to start."""
        self.app.call_from_thread(self.hotkey_start_recording)

    def hotkey_on_stop(self) -> None:
        """Called from the hotkey thread when the hotkey is pressed to stop."""
        self.app.call_from_thread(self.hotkey_stop_recording)

    def hotkey_start_recording(self) -> None:
        """Start recording triggered by global hotkey (runs on main thread)."""
        if self.app._recording or self.app._transcribing:
            return
        if not self.app._model_ready:
            self.overlay_set("error")
            return
        # Switch to record tab so the user can see what's happening
        with contextlib.suppress(Exception):
            self.app.query_one("#tabs", TabbedContent).active = "tab-record"
        self.overlay_set("recording")
        # Delegate to recording handler
        from voicepad.tui.handlers.recording_handler import RecordingHandler

        handler = RecordingHandler(self.app)
        handler.start_recording()

    def hotkey_stop_recording(self) -> None:
        """Stop recording triggered by global hotkey (runs on main thread)."""
        if not self.app._recording:
            return
        self.app._hotkey_pending_copy = True  # flag to auto-copy after transcription
        self.overlay_set("transcribing")
        # Delegate to recording handler
        from voicepad.tui.handlers.recording_handler import RecordingHandler

        handler = RecordingHandler(self.app)
        handler.stop_recording()

    def overlay_set(self, state: str) -> None:
        """Update the floating overlay state if it exists."""
        if self.app._overlay is not None:
            with contextlib.suppress(Exception):
                self.app._overlay.set_state(state)  # type: ignore[union-attr]
