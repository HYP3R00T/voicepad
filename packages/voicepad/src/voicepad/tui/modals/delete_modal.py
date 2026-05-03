"""Delete confirmation modal for VoicePad TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from voicepad.tui.components import VoiceButton

if TYPE_CHECKING:
    from textual.app import ComposeResult


class DeleteConfirmModal(ModalScreen[bool]):
    """Confirmation dialog before deleting a recording."""

    BINDINGS = [
        Binding("escape", "dismiss_false", "Cancel", show=False),
        Binding("n", "dismiss_false", "No", show=False),
    ]

    def __init__(self, entry_name: str) -> None:
        super().__init__()
        self._entry_name = entry_name

    def compose(self) -> ComposeResult:
        with Static(id="delete-dialog"):
            yield Static("󰆴  Delete recording?", id="delete-title")
            yield Static(f"[dim]{self._entry_name}[/]", id="delete-name")
            yield Static(
                "This will permanently delete the WAV file\nand its transcription markdown.",
                id="delete-body",
            )
            with Static(id="delete-nav"):
                yield VoiceButton("Cancel", role="default", id="delete-cancel")
                yield Static("", id="delete-spacer")
                yield VoiceButton("Delete", role="danger", id="delete-confirm")

    def action_dismiss_false(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#delete-cancel")
    def on_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#delete-confirm")
    def on_confirm(self) -> None:
        self.dismiss(True)
