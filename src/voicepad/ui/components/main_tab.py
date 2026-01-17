"""Main tab for the Voicepad TUI application."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static


class MainTab(Container):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Welcome to Voicepad", classes="section-title")
            yield Static(
                "This is the main tab. Additional content and features will be added here.",
                classes="section-content",
            )

            yield Static("Status", classes="section-title")
            yield Static("Ready to use.", classes="section-content")
