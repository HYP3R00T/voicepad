from __future__ import annotations

import contextlib
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Footer, Header, Static
from voicepad_core.audio import MicrophoneStream
from voicepad_core.pipeline import FileTranscriptionResult, GrowingTranscriptionJob

from voicepad.config import AppConfig, load_config
from voicepad.output import persist_markdown
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.control import ControlServer
from voicepad.tui.utils.clipboard import copy_to_clipboard


class VoicePadApp(App[None]):
    """Small resident-model dictation interface."""

    TITLE = "VoicePad"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("space", "toggle_recording", "Record / stop"),
        Binding("c", "copy", "Copy"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, config: AppConfig | None = None, runtime: ApplicationRuntime | None = None) -> None:
        super().__init__()
        self.config = config or load_config()
        self.runtime = runtime or ApplicationRuntime(self.config)
        self._state = "loading"
        self._microphone: MicrophoneStream | None = None
        self._job: GrowingTranscriptionJob | None = None
        self._last_result: FileTranscriptionResult | None = None
        self._last_wav: Path | None = None
        self._control = ControlServer(lambda: self.call_from_thread(self.action_toggle_recording))

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Loading verified NVIDIA deployment…", id="status")
        yield Static("Your transcription will appear here.", id="transcript")
        yield Static("", id="metadata")
        yield Button("Record", id="record", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._control.start()
        self._activate()

    def on_unmount(self) -> None:
        self._control.stop()
        if self._microphone is not None and self._microphone.is_recording:
            with contextlib.suppress(Exception):
                self._microphone.stop()
        if self._job is not None:
            self._job.cancel()
            with contextlib.suppress(Exception):
                self._job.finish(timeout=30)
        with contextlib.suppress(Exception):
            self.runtime.close()

    @work(thread=True, exclusive=True, group="activation")
    def _activate(self) -> None:
        try:
            active = self.runtime.activate()
        except Exception as error:
            self.call_from_thread(self._set_error, f"Activation failed: {error}")
            return
        self.call_from_thread(self._set_ready, active.device_name)

    def _set_ready(self, device_name: str) -> None:
        self._state = "ready"
        self.query_one("#status", Static).update(f"Ready — {device_name}")
        self.query_one("#record", Button).disabled = False

    def _set_error(self, message: str) -> None:
        self._state = "error"
        self.query_one("#status", Static).update(message)
        self.query_one("#record", Button).disabled = True

    @on(Button.Pressed, "#record")
    def record_pressed(self) -> None:
        self.action_toggle_recording()

    def action_toggle_recording(self) -> None:
        if self._state == "ready":
            self._state = "starting"
            self.query_one("#status", Static).update("Starting recording…")
            self._start_recording()
        elif self._state == "recording":
            self._state = "transcribing"
            self.query_one("#status", Static).update("Finalizing WAV and transcription…")
            self.query_one("#record", Button).disabled = True
            self._stop_recording()

    @work(thread=True, exclusive=True, group="recording-start")
    def _start_recording(self) -> None:
        try:
            microphone, job = self.runtime.start_recording()
        except Exception as error:
            self.call_from_thread(self._set_error, f"Recording failed: {error}")
            return
        self.call_from_thread(self._recording_started, microphone, job)

    def _recording_started(self, microphone: MicrophoneStream, job: GrowingTranscriptionJob) -> None:
        self._microphone = microphone
        self._job = job
        self._state = "recording"
        self.query_one("#status", Static).update("Recording… press Space to stop")
        self.query_one("#record", Button).label = "Stop"

    @work(thread=True, exclusive=True, group="recording-stop")
    def _stop_recording(self) -> None:
        microphone = self._microphone
        job = self._job
        if microphone is None or job is None:
            self.call_from_thread(self._set_error, "Recording state is incomplete.")
            return
        try:
            artifact, result = self.runtime.stop_recording(microphone, job)
            markdown = persist_markdown(artifact.path, result, self.config.markdown_path)
            if result.complete and result.text and self.config.copy_complete_text:
                copy_to_clipboard(result.text)
        except Exception as error:
            self.call_from_thread(self._set_error, f"Transcription failed: {error}")
            return
        self.call_from_thread(self._recording_finished, artifact.path, markdown, result)

    def _recording_finished(
        self,
        wav_path: Path,
        markdown_path: Path,
        result: FileTranscriptionResult,
    ) -> None:
        self._microphone = None
        self._job = None
        self._last_wav = wav_path
        self._last_result = result
        self.query_one("#transcript", Static).update(result.text or "No speech detected.")
        self.query_one("#metadata", Static).update(
            f"{result.duration_seconds:.1f}s · {len(result.chunks)} chunks · "
            f"{'complete' if result.complete else 'incomplete'} · {markdown_path.name}"
        )
        self.query_one("#record", Button).label = "Record"
        self.query_one("#record", Button).disabled = False
        if result.complete:
            self._set_ready(result.deployment.device_name)
        else:
            self._state = "ready"
            self.query_one("#status", Static).update("Incomplete result saved; audio is preserved.")

    def action_copy(self) -> None:
        if self._last_result is not None and self._last_result.complete and self._last_result.text:
            if copy_to_clipboard(self._last_result.text):
                self.notify("Copied transcription")
            else:
                self.notify("Clipboard is unavailable", severity="warning")


def run(config: AppConfig | None = None) -> None:
    VoicePadApp(config).run()
