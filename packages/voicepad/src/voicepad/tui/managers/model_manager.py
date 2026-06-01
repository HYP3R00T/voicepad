"""Model warming and status management for VoicePad TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.widgets import Label

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp
    from voicepad.tui.workers import ModelWarmResult

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model warming, status updates, and model reloading."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app

    def on_model_ready(self, result: ModelWarmResult) -> None:
        """Handle model warm-up completion."""
        self.app._warm_result = result
        model_label = self.app.query_one("#header-model", Label)

        if result.error:
            self.set_status("error", f"model error: {result.error}")
            return

        fallback = "  cpu fallback" if result.fallback else ""
        model_label.update(
            f"[dim]model:[/] {self.app.config.transcription_model}  [dim]device:[/] {result.device}{fallback}"
        )
        self.set_status("ready", "ready")
        self.app._model_ready = True

        # Start the global hotkey listener now that the model is ready
        # and the Textual event loop is fully running.
        if self.app._hotkey_listener is None:
            self.app._start_hotkey_listener()

    def reload_model(self) -> None:
        """Re-download (if needed) and reload the current model."""
        if self.app._recording or self.app._transcribing:
            return
        from voicepad_core import _model_cache

        _model_cache.clear()
        self.app._model_ready = False
        self.set_status("transcribing", "reloading model…")
        self.app.query_one("#header-model", Label).update("[dim]model:[/] loading…")
        self.app._warm_model_worker()

    def set_status(self, state: str, message: str) -> None:
        """Update the status label with icon and message."""
        dots = {"ready": "", "recording": "\U000f044a", "transcribing": "\U000f051f", "error": "\U000f0159"}
        dot = dots.get(state, "\U000f051f")
        label = self.app.query_one("#status", Label)
        label.remove_class("ready", "recording", "transcribing", "error")
        if state:
            label.add_class(state)
        label.update(f"{dot}  {message}")
