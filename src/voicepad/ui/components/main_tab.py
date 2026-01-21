"""Main tab for the Voicepad TUI application."""

from textual.app import ComposeResult
from textual.containers import Container

from voicepad.ui.components.recording_panel import RecordingPanel


class MainTab(Container):
    def compose(self) -> ComposeResult:
        yield RecordingPanel(id="rec_panel")
