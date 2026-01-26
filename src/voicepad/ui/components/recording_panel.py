"""Recording panel with card-based layout."""

import threading
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Label, RichLog, Select, Static

from voicepad.config import get_config
from voicepad.voice import capture_audio_background, get_input_devices, get_recording_path


class RecordingPanel(Container):
    """Recording panel with compact card layout."""

    DEFAULT_CSS = """
    RecordingPanel {
        layout: vertical;
        padding: 1;
    }

    /* Top row with cards */
    #top_row {
        height: auto;
        margin-bottom: 1;
    }

    /* Device card */
    #device_card {
        width: 1fr;
        height: auto;
        border: round $primary;
        padding: 0 1;
        margin-right: 1;
    }

    #device_card Select {
        width: 100%;
        border: none;
        background: transparent;
    }

    #device_card SelectCurrent {
        border: none;
        padding: 0;
        background: transparent;
    }

    /* Status card */
    #status_card {
        width: auto;
        min-width: 14;
        height: auto;
        border: round $primary;
        padding: 0 1;
    }

    #status_row {
        height: 1;
        width: auto;
    }

    .status-dot {
        width: 2;
    }

    .dot-ready {
        color: $success;
    }

    .dot-recording {
        color: $error;
    }

    /* Control card */
    #control_card {
        width: auto;
        min-width: 10;
        height: auto;
        border: round $primary;
        padding: 0;
        margin-left: 1;
    }

    #record_btn {
        width: 100%;
        border: none;
        background: transparent;
    }

    #record_btn:hover {
        background: $boost;
    }

    /* Log section */
    #log_section {
        border: round $primary;
        height: 1fr;
    }

    #log {
        height: 100%;
        background: transparent;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.current_path: Path | None = None
        self.is_recording = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="top_row"):
            # Device selection card
            device_card = Container(id="device_card")
            device_card.border_title = "Device"
            with device_card:
                yield Select([], id="mic_select", prompt="Select...")

            # Status card
            status_card = Container(id="status_card")
            status_card.border_title = "Status"
            with status_card, Horizontal(id="status_row"):
                yield Static("●", id="status_dot", classes="status-dot dot-ready")
                yield Label("Ready", id="status_text")

            # Control card
            control_card = Container(id="control_card")
            control_card.border_title = "Control"
            with control_card:
                yield Button("● Rec", id="record_btn")

        # Log section
        log_section = Container(id="log_section")
        log_section.border_title = "Log"
        with log_section:
            yield RichLog(id="log", wrap=True, markup=True)

    def on_mount(self) -> None:
        devices = get_input_devices()
        select = self.query_one("#mic_select", Select)
        if devices:
            select.set_options([(d.name, d.index) for d in devices])
            select.value = devices[0].index

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "record_btn":
            self.action_toggle_recording()

    def action_toggle_recording(self) -> None:
        if self.is_recording:
            self._stop()
        else:
            self._start()

    def action_quit(self) -> None:
        if self.is_recording:
            self._stop()
        self.app.exit()

    def _update_status(self, recording: bool) -> None:
        dot = self.query_one("#status_dot", Static)
        text = self.query_one("#status_text", Label)
        btn = self.query_one("#record_btn", Button)

        if recording:
            dot.remove_class("dot-ready")
            dot.add_class("dot-recording")
            text.update("Recording")
            btn.label = "■ Stop"
        else:
            dot.remove_class("dot-recording")
            dot.add_class("dot-ready")
            text.update("Ready")
            btn.label = "● Rec"

    def _start(self) -> None:
        log = self.query_one("#log", RichLog)
        select = self.query_one("#mic_select", Select)

        if select.value is None or select.value == Select.BLANK:
            log.write("[red]● Select a microphone first[/red]")
            return

        device_index: int = select.value  # type: ignore[assignment]

        config = get_config()
        config.recordings_path.mkdir(parents=True, exist_ok=True)
        self.current_path = get_recording_path(config.recordings_path)

        self.stop_event.clear()
        self.capture_thread = capture_audio_background(device_index, self.current_path, self.stop_event)
        self.is_recording = True
        self._update_status(True)
        log.write(f"[green]● Recording started[/green] → {self.current_path.name}")

    def _stop(self) -> None:
        log = self.query_one("#log", RichLog)

        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None

        self.is_recording = False
        self._update_status(False)

        if self.current_path and self.current_path.exists():
            kb = self.current_path.stat().st_size / 1024
            log.write(f"[green]● Saved[/green] {self.current_path.name} ({kb:.1f} KB)")
        self.current_path = None
