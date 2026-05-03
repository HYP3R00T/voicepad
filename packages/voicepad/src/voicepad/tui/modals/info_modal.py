"""Info modal for VoicePad TUI."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Link, Static

from voicepad.tui.components import VoiceButton

if TYPE_CHECKING:
    from textual.app import ComposeResult

try:
    _APP_VERSION = f"v{_pkg_version('voicepad')}"
except Exception:
    _APP_VERSION = "dev"


class InfoModal(ModalScreen[None]):
    """App info, version, and sponsor links."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("i", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:

        with Static(id="info-dialog"):
            yield Static("󰍬  VoicePad", id="info-title")
            yield Static(
                "Your voice, your data.\nTranscription that never leaves your machine.",
                id="info-subtitle",
            )
            with Static(id="info-guarantees"):
                yield Static("󰒍  Fully local processing", classes="guarantee-line")
                yield Static("󰌨  GPU-accelerated transcription", classes="guarantee-line")
                yield Static("󰍹  No cloud. No tracking. No data leaks.", classes="guarantee-line")
            yield Static(
                "Built with 󰋑 using Python, Textual, and Whisper",
                id="info-philosophy",
            )
            yield Static("", id="info-divider")
            yield Static("Support the project", id="info-sponsor-title")
            with Static(id="info-links"):
                yield Link(
                    "󰊤  Star on GitHub",
                    url="https://github.com/HYP3R00T/voicepad",
                    id="github-link",
                )
                yield Static("  ", classes="link-separator")
                yield Link(
                    "󰋑  Sponsor Me",
                    url="https://github.com/sponsors/HYP3R00T",
                    id="sponsor-link",
                )
            yield Static(
                "Privacy-first tools grow through community support.",
                id="info-microcopy",
            )
            yield Static(
                f"{_APP_VERSION}  •  Rajesh Das (HYP3R00T)  •  MIT License",
                id="info-meta",
            )
            yield Static("", id="info-divider2")
            yield VoiceButton("Close", role="default", id="info-close-btn")

    def on_button_pressed(self, event) -> None:  # type: ignore[override]
        self.dismiss()
