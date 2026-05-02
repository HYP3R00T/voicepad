"""Header component with status and model info."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Label, Static


class HeaderWidget(Static):
    """App header with title, version, status, and model info."""

    def __init__(self, version: str = "dev") -> None:
        super().__init__(id="header")
        self._version = version

    def compose(self) -> ComposeResult:
        yield Label("VoicePad", id="header-title")
        yield Label(self._version, id="header-version")
        yield Label("\U000f051f  initialising", id="status")
        yield Label("loading…", id="header-model")

    def set_status(self, state: str, message: str) -> None:
        """Update status label with icon and message."""
        dots = {
            "ready": "",
            "recording": "\U000f044a",
            "transcribing": "\U000f051f",
            "error": "\U000f0159",
        }
        dot = dots.get(state, "\U000f051f")
        label = self.query_one("#status", Label)
        label.remove_class("ready", "recording", "transcribing", "error")
        if state:
            label.add_class(state)
        label.update(f"{dot}  {message}")

    def set_model_info(self, model: str, device: str, fallback: bool = False) -> None:
        """Update model info label."""
        fallback_text = "  cpu fallback" if fallback else ""
        self.query_one("#header-model", Label).update(
            f"[dim]model:[/] {model}  [dim]device:[/] {device}{fallback_text}"
        )
