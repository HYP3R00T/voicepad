"""VoicePad Textual TUI application."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, Static
from voicepad_core import AudioRecorder, AudioRecorderError, get_config
from voicepad_core.config import Config

from voicepad.tui.workers import ModelWarmResult, RecordingSession, TranscriptionJob

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class SessionEntry:
    """One completed recording + transcription in the session history."""

    index: int
    wav_path: Path | None
    duration_s: float
    text: str
    latency_ms: float
    device: str
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """Top status line showing model, device, and current state."""

    DEFAULT_CSS = """
    StatusBar {
        background: $panel;
        color: $text-muted;
        padding: 0 2;
        height: 1;
    }
    StatusBar.ready   { color: $success; }
    StatusBar.loading { color: $warning; }
    StatusBar.error   { color: $error; }
    """

    def update_status(self, text: str, state: str = "") -> None:
        self.remove_class("ready", "loading", "error")
        if state:
            self.add_class(state)
        self.update(text)


class TranscriptionView(Static):
    """Displays the most recent transcription result."""

    DEFAULT_CSS = """
    TranscriptionView {
        border: round $primary;
        padding: 1 2;
        height: auto;
        min-height: 5;
        margin: 1 0;
    }
    TranscriptionView .placeholder {
        color: $text-disabled;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("No transcription yet", classes="placeholder", id="transcription-text")

    def show(self, text: str, latency_ms: float, device: str) -> None:
        label = self.query_one("#transcription-text", Label)
        label.remove_class("placeholder")
        label.update(text or "(no speech detected)")
        self.border_subtitle = f"{device}  ·  {latency_ms:.0f}ms"


class HistoryList(ListView):
    """Scrollable list of past session entries."""

    DEFAULT_CSS = """
    HistoryList {
        height: 1fr;
        border: round $surface;
    }
    HistoryList > ListItem {
        padding: 0 1;
    }
    """

    def add_entry(self, entry: SessionEntry) -> None:
        duration = f"{entry.duration_s:.1f}s"
        name = entry.wav_path.stem if entry.wav_path else f"clip-{entry.index}"
        preview = entry.text[:60] + "…" if len(entry.text) > 60 else entry.text
        label = f"[dim]{entry.timestamp}[/]  {name}  [dim]{duration}[/]\n  {preview or '(empty)'}"
        self.append(ListItem(Label(label)))
        self.scroll_end(animate=False)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class VoicePadApp(App[None]):
    """VoicePad — local dictation with Whisper."""

    TITLE = "VoicePad"
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        layout: vertical;
        padding: 0 1;
    }
    #record-btn {
        width: 100%;
        margin: 1 0;
    }
    #record-btn.recording {
        background: $error;
        color: $text;
    }
    #record-btn.transcribing {
        background: $warning;
        color: $text;
    }
    #timer {
        text-align: center;
        color: $text-muted;
        height: 1;
    }
    #history-label {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_recording", "Record", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    # Reactive state
    _model_ready: reactive[bool] = reactive(False)
    _recording: reactive[bool] = reactive(False)
    _transcribing: reactive[bool] = reactive(False)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._session: RecordingSession | None = None
        self._entries: list[SessionEntry] = []
        self._record_start: float = 0.0
        self._timer_thread: threading.Thread | None = None
        self._warm_result: ModelWarmResult | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main"):
            yield StatusBar("Loading model…", id="status")
            yield Button("● START RECORDING  [Space]", id="record-btn", variant="success")
            yield Label("", id="timer")
            yield TranscriptionView(id="transcription")
            yield Label("Session history", id="history-label")
            yield HistoryList(id="history")
        yield Footer()

    # ------------------------------------------------------------------
    # Startup — warm model in background
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.query_one("#record-btn", Button).disabled = True
        self._warm_model_worker()

    @work(thread=True, exclusive=True, name="model-warm")
    def _warm_model_worker(self) -> None:
        from voicepad.tui.workers import warm_model

        result = warm_model(self.config)
        self.call_from_thread(self._on_model_ready, result)

    def _on_model_ready(self, result: ModelWarmResult) -> None:
        self._warm_result = result
        status = self.query_one("#status", StatusBar)
        btn = self.query_one("#record-btn", Button)

        if result.error:
            status.update_status(
                f"Model error: {result.error}",
                state="error",
            )
            return

        device_label = f"{result.device} ({result.compute_type})"
        fallback_note = "  [CPU fallback]" if result.fallback else ""
        status.update_status(
            f"Model: {self.config.transcription_model}  ·  {device_label}{fallback_note}  ·  Ready",
            state="ready",
        )
        btn.disabled = False
        self._model_ready = True

    # ------------------------------------------------------------------
    # Record / stop toggle
    # ------------------------------------------------------------------

    def action_toggle_recording(self) -> None:
        if not self._model_ready:
            return
        if self._transcribing:
            return
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    @on(Button.Pressed, "#record-btn")
    def on_record_btn(self) -> None:
        self.action_toggle_recording()

    def _start_recording(self) -> None:
        self._session = RecordingSession(config=self.config)
        btn = self.query_one("#record-btn", Button)
        status = self.query_one("#status", StatusBar)

        try:
            self._session.start()
        except AudioRecorderError as e:
            status.update_status(f"Mic error: {e}", state="error")
            return

        self._recording = True
        self._record_start = time.monotonic()
        btn.label = "■ STOP RECORDING  [Space]"
        btn.add_class("recording")
        btn.variant = "error"
        status.update_status("Recording…", state="loading")
        self._start_timer()

    def _stop_recording(self) -> None:
        if self._session is None:
            return

        self._recording = False
        self._stop_timer()

        btn = self.query_one("#record-btn", Button)
        btn.disabled = True
        btn.remove_class("recording")
        btn.label = "⟳ Transcribing…"
        btn.add_class("transcribing")
        btn.variant = "warning"

        status = self.query_one("#status", StatusBar)
        status.update_status("Transcribing…", state="loading")

        try:
            audio = self._session.stop()
        except AudioRecorderError as e:
            status.update_status(f"Stop error: {e}", state="error")
            self._reset_button()
            return

        self._transcribing = True
        self._transcribe_worker(audio)

    # ------------------------------------------------------------------
    # Transcription worker
    # ------------------------------------------------------------------

    @work(thread=True, name="transcribe")
    def _transcribe_worker(self, audio: np.ndarray) -> None:
        job = TranscriptionJob(audio=audio, config=self.config)
        result = job.run()
        self.call_from_thread(self._on_transcription_done, audio, result, job.error)

    def _on_transcription_done(
        self,
        audio: np.ndarray,
        result,
        error: str | None,
    ) -> None:
        self._transcribing = False
        status = self.query_one("#status", StatusBar)
        transcription_view = self.query_one("#transcription", TranscriptionView)
        history = self.query_one("#history", HistoryList)

        if error:
            status.update_status(f"Error: {error}", state="error")
            self._reset_button()
            return

        # Save WAV + markdown
        wav_path: Path | None = None
        if self._session and result:
            recorder_ref: AudioRecorder | None = self._session._recorder
            if recorder_ref is not None:
                wav_path = recorder_ref.make_wav_path()
                try:
                    recorder_ref.save_wav(audio, wav_path)
                    # Save markdown
                    md_path = self.config.markdown_path / f"{wav_path.stem}.md"
                    self.config.markdown_path.mkdir(parents=True, exist_ok=True)
                    md_path.write_text(
                        _format_markdown(wav_path, result),
                        encoding="utf-8",
                    )
                except Exception:
                    wav_path = None

        # Update UI
        if result:
            transcription_view.show(result.text, result.latency_ms, result.device)
            device_label = f"{result.device} ({result.compute_type})"
            fallback_note = "  [CPU fallback]" if result.fallback_to_cpu else ""
            status.update_status(
                f"Model: {self.config.transcription_model}  ·  {device_label}{fallback_note}  ·  Ready",
                state="ready",
            )

            entry = SessionEntry(
                index=len(self._entries),
                wav_path=wav_path,
                duration_s=result.duration_s,
                text=result.text,
                latency_ms=result.latency_ms,
                device=result.device,
            )
            self._entries.append(entry)
            history.add_entry(entry)

        self._reset_button()

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        self._timer_thread = None
        self.call_from_thread_safe(self.query_one("#timer", Label).update, "")

    def _timer_loop(self) -> None:
        while self._recording:
            elapsed = time.monotonic() - self._record_start
            self.call_from_thread(
                self.query_one("#timer", Label).update,
                f"⏱  {elapsed:.1f}s",
            )
            time.sleep(0.1)

    def call_from_thread_safe(self, fn, *args) -> None:  # noqa: ANN001
        """No-op wrapper — timer stop is best-effort."""
        with contextlib.suppress(Exception):
            self.call_from_thread(fn, *args)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_button(self) -> None:
        btn = self.query_one("#record-btn", Button)
        btn.remove_class("recording", "transcribing")
        btn.variant = "success"
        btn.label = "● START RECORDING  [Space]"
        btn.disabled = not self._model_ready


# ---------------------------------------------------------------------------
# Markdown formatter (reused from CLI)
# ---------------------------------------------------------------------------


def _format_markdown(wav_path: Path, result) -> str:
    lines = [
        "# Transcription",
        "",
        f"**File:** `{wav_path.name}`",
        f"**Model:** {result.device} / {result.compute_type}",
        f"**Language:** {result.language} ({result.language_probability * 100:.1f}%)",
        f"**Duration:** {result.duration_s:.1f}s",
        f"**Latency:** {result.latency_ms:.0f}ms",
        "",
        "---",
        "",
        "## Text",
        "",
        result.text or "*(no speech detected)*",
        "",
    ]
    if result.segments:
        lines += ["## Segments", ""]
        for seg in result.segments:
            lines.append(f"- `[{seg.start:.1f}s → {seg.end:.1f}s]` {seg.text}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the VoicePad TUI."""
    config = get_config()
    app = VoicePadApp(config)
    app.run()
