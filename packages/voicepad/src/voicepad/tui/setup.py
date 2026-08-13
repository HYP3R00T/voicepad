from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Label, ProgressBar, Static

from voicepad.tui.components import VoiceButton

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp


class SetupModal(ModalScreen[None]):
    """Explain incomplete local setup and obtain consent before downloading."""

    BINDINGS = [Binding("q", "app.quit", "Quit")]

    def __init__(self, *, config_missing: bool, artifacts_missing: bool | None, artifact_path: Path) -> None:
        super().__init__()
        self.config_missing = config_missing
        self.artifacts_missing = artifacts_missing
        self.artifact_path = artifact_path

    def compose(self) -> ComposeResult:
        if self.config_missing and self.artifacts_missing is None:
            reason = "VoicePad has no saved configuration yet."
        elif self.config_missing and self.artifacts_missing:
            reason = "VoicePad has no saved configuration yet, and its required model files are not ready."
        elif self.config_missing:
            reason = "VoicePad has no saved configuration yet. Its existing model files are ready."
        else:
            reason = "VoicePad's required model files are missing or could not be verified."
        if self.artifacts_missing is None:
            explanation = (
                "VoicePad will save its default settings, check for existing NVIDIA Parakeet and Silero VAD "
                "files, and download about 2.4 GiB only if they are missing."
            )
            button_label = "Set up VoicePad"
        elif self.artifacts_missing:
            explanation = (
                "VoicePad will download and verify NVIDIA Parakeet and Silero VAD "
                "for private, local transcription. The download is about 2.4 GiB."
            )
            button_label = "Download and continue"
        else:
            explanation = "VoicePad will save the default settings shown in the Settings tab before it starts."
            button_label = "Save and continue"
        with Static(id="setup-dialog"):
            yield Label("Welcome to VoicePad", id="setup-title")
            yield Label(reason, id="setup-reason")
            yield Static(explanation, id="setup-explanation")
            yield Label(f"Model storage: {self.artifact_path}", id="setup-path")
            yield ProgressBar(id="setup-progress", show_eta=False, show_percentage=False)
            yield Label(
                "Nothing will be downloaded until you continue."
                if self.artifacts_missing is not False
                else "Ready to continue.",
                id="setup-status",
            )
            with Horizontal(id="setup-actions"):
                yield VoiceButton(button_label, role="primary", id="setup-continue")

    def on_mount(self) -> None:
        self.query_one("#setup-progress", ProgressBar).display = False

    @on(VoiceButton.Pressed, "#setup-continue")
    def continue_setup(self) -> None:
        self.query_one("#setup-actions", Horizontal).display = False
        progress = self.query_one("#setup-progress", ProgressBar)
        progress.display = True
        progress.update(total=None)
        self.query_one("#setup-status", Label).update("Preparing local models…")
        cast("VoicePadApp", self.app).begin_setup(self)

    def update_progress(self, completed: int, total: int) -> None:
        self.query_one("#setup-progress", ProgressBar).update(total=total, progress=completed)
        completed_gib = completed / 1024**3
        total_gib = total / 1024**3
        self.query_one("#setup-status", Label).update(
            f"Downloaded and verified {completed_gib:.2f} of {total_gib:.2f} GiB…"
        )

    def show_error(self, error: Exception) -> None:
        self.query_one("#setup-progress", ProgressBar).display = False
        self.query_one("#setup-status", Label).update(f"Setup failed: {error}")
        self.query_one("#setup-actions", Horizontal).display = True
        button = self.query_one("#setup-continue", VoiceButton)
        button.label = "Retry setup"
        button.disabled = False
