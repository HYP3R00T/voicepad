"""Minimal Textual application for VoicePad."""

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, ListItem, ListView, Static

from voicepad.microphones import get_microphones
from voicepad.recorder import record_audio, stop_recording


class VoicePadApp(App):
    """Minimal VoicePad Textual app."""

    TITLE = "VoicePad"
    THEME = "catppuccin-mocha"
    CSS_PATH = "styles.css"

    def __init__(self):
        """Initialize the app."""
        super().__init__()
        self.microphones = []
        self.selected_microphone = None
        self.is_recording = False

    def compose(self) -> ComposeResult:
        """Create widgets."""
        self.microphones = get_microphones()

        with Vertical():
            yield Static("Available Microphones:", id="header")
            list_view = ListView(*[ListItem(Static(f"{mic['name']}")) for mic in self.microphones], id="mic_list")
            yield list_view

            with Horizontal():
                yield Input(placeholder="Recording duration (seconds)", id="duration_input", value="5")

            with Horizontal(id="buttons_container"):
                yield Button("Record", id="record_btn", variant="primary")
                yield Button("Stop", id="stop_btn")
                yield Button("Exit", id="exit_btn", variant="error")

            yield Static("", id="status_message")

    def on_mount(self) -> None:
        """Handle app mount."""
        self.title = "VoicePad - Microphone Recording"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle microphone selection."""
        list_view = self.query_one("#mic_list", ListView)
        index = list_view.index
        if index is not None and index < len(self.microphones):
            self.selected_microphone = self.microphones[index]
            self.update_status(f"Selected: {self.selected_microphone['name']}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "record_btn":
            self._start_recording()
        elif event.button.id == "stop_btn":
            self._stop_recording()
        elif event.button.id == "exit_btn":
            self.exit()

    def _start_recording(self) -> None:
        """Start recording audio."""
        if not self.selected_microphone:
            self.update_status("Please select a microphone first")
            return

        try:
            duration_input = self.query_one("#duration_input", Input)
            duration = float(duration_input.value)
        except ValueError:
            self.update_status("Invalid duration. Please enter a number.")
            return

        self.is_recording = True
        self.update_status(f"Recording for {duration} seconds...")

        # Run recording in background
        self.run_worker(self._record_worker(duration))

    async def _record_worker(self, duration: float) -> None:
        """Worker to handle recording in background."""
        mic = self.selected_microphone
        loop = asyncio.get_event_loop()
        filepath = await loop.run_in_executor(
            None,
            record_audio,
            mic["index"],
            mic["channels"],
            mic["sample_rate"],
            duration,
        )

        if filepath:
            self.update_status(f"Recording saved to: {filepath}")
        else:
            self.update_status("Recording failed")

        self.is_recording = False

    async def _async_record(
        self,
        device_index: int,
        channels: int,
        sample_rate: int,
        duration: float,
    ) -> str | None:
        """Asynchronously record audio."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            record_audio,
            device_index,
            channels,
            sample_rate,
            duration,
        )

    def _stop_recording(self) -> None:
        """Stop recording."""
        if self.is_recording:
            stop_recording()
            self.update_status("Recording stopped")
            self.is_recording = False
        else:
            self.update_status("No recording in progress")

    def update_status(self, message: str) -> None:
        """Update status message."""
        status = self.query_one("#status_message", Static)
        status.update(message)


app = VoicePadApp()
