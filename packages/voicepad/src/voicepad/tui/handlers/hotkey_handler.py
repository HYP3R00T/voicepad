"""Hotkey handler for VoicePad TUI."""

from __future__ import annotations

import logging
import platform
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
        """Start the status overlay and the native Windows hotkey listener."""
        hotkey = getattr(self.app.config, "global_hotkey", "")
        is_windows = platform.system() == "Windows"
        if is_windows and not hotkey:
            return

        try:
            from voicepad.tui.overlay import StatusOverlay

            if self.app._overlay is None:
                self.app._overlay = StatusOverlay(theme=getattr(self.app.tui_config, "theme", "tokyo-night"))
                self.app._overlay.start()
        except Exception as error:
            logger.warning("Could not start the hotkey status overlay: %s", error)

        if not is_windows:
            logger.info(
                "Native global hotkey registration is unavailable; use the desktop shortcut command 'voicepad toggle'"
            )
            return

        try:
            from voicepad.tui.hotkey import GlobalHotkeyListener

            listener = GlobalHotkeyListener(
                hotkey=hotkey,
                on_toggle=self.hotkey_on_toggle,
            )
            listener.start()
            self.app._hotkey_listener = listener
        except Exception as error:
            self.app._hotkey_listener = None
            logger.warning("Could not register global hotkey '%s': %s", hotkey, error)

    def hotkey_on_toggle(self) -> None:
        """Dispatch a hotkey toggle request to the Textual thread."""
        self.app.call_from_thread(self.hotkey_toggle_recording)

    def hotkey_toggle_recording(self) -> None:
        """Toggle recording from the Textual thread."""
        action = "stop" if self.app._recording else "start"
        logger.info("Desktop shortcut toggle received: action=%s", action)
        if self.app._recording:
            self.hotkey_stop_recording()
        else:
            self.hotkey_start_recording()

    def hotkey_start_recording(self) -> None:
        """Start recording triggered by global hotkey (runs on main thread)."""
        if self.app._recording or self.app._transcribing:
            logger.debug(
                "Ignoring recording toggle: recording=%s transcribing=%s",
                self.app._recording,
                self.app._transcribing,
            )
            return
        if not self.app._model_ready:
            logger.warning("Ignoring recording toggle because the transcription model is not ready")
            self.overlay_set("error")
            return
        # Switch to record tab so the user can see what's happening
        try:
            self.app.query_one("#tabs", TabbedContent).active = "tab-record"
        except Exception as error:
            logger.warning("Could not show the recording tab for a desktop shortcut: %s", error)
        self.overlay_set("recording")
        self.app._recording_handler.start_recording()

    def hotkey_stop_recording(self) -> None:
        """Stop recording triggered by global hotkey (runs on main thread)."""
        if not self.app._recording:
            return
        self.overlay_set("transcribing")
        self.app._recording_handler.stop_recording()

    def overlay_set(self, state: str) -> None:
        """Update the floating overlay state if it exists."""
        if self.app._overlay is not None:
            try:
                self.app._overlay.set_state(state)  # type: ignore[union-attr]
            except Exception as error:
                logger.warning("Could not update the hotkey status overlay to '%s': %s", state, error)
