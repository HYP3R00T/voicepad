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
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Button, Footer, Label, ListItem, ListView, Static
from voicepad_core import AudioRecorder, AudioRecorderError, get_config
from voicepad_core.config import Config

from voicepad.tui.workers import ModelWarmResult, RecordingSession, TranscriptionJob

# ---------------------------------------------------------------------------
# Theme — registered on mount, uses Textual's semantic variable system
# ---------------------------------------------------------------------------

VOICEPAD_THEME = Theme(
    name="voicepad",
    primary="#58a6ff",
    secondary="#bc8cff",
    accent="#58a6ff",
    foreground="#e6edf3",
    background="#0d1117",
    surface="#161b22",
    panel="#21262d",
    success="#3fb950",
    warning="#d29922",
    error="#f85149",
    dark=True,
    variables={
        "border-blurred": "#30363d",
        "footer-key-foreground": "#58a6ff",
        "footer-background": "#161b22",
        "footer-foreground": "#7d8590",
        "block-cursor-text-style": "none",
    },
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class SessionEntry:
    index: int
    wav_path: Path | None
    duration_s: float
    text: str
    latency_ms: float
    device: str
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class VoicePadApp(App[None]):
    """VoicePad — local dictation with Whisper."""

    TITLE = "VoicePad"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("space", "toggle_recording", "Record / Stop", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

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
        # Minimal custom header — one line, no chrome
        with Static(id="header"):
            yield Label("voicepad", id="header-title")
            yield Label("loading…", id="header-model")

        with Static(id="body"):
            yield Label("○  initialising", id="status")
            with Static(id="controls"):
                yield Button("○  start recording", id="record-btn", disabled=True)
                yield Label("", id="timer")

            # Transcription panel
            tx = Static(id="transcription")
            tx.border_title = "transcription"
            yield tx

            # History panel
            hist = Static(id="history")
            hist.border_title = "history"
            yield hist

        yield Footer()

    def on_mount(self) -> None:
        # Add children to panels after mount (Static doesn't support compose)
        tx = self.query_one("#transcription", Static)
        tx.mount(Label("speak and press space to begin…", id="tx-text", classes="placeholder"))
        tx.mount(Label("", id="tx-meta"))

        hist = self.query_one("#history", Static)
        hist.mount(Label("no recordings yet this session", id="empty-hint"))
        hist.mount(ListView(id="history-list"))

        self.register_theme(VOICEPAD_THEME)
        self.theme = "voicepad"
        self._warm_model_worker()

    # ------------------------------------------------------------------
    # Model warm-up
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=True, name="model-warm")
    def _warm_model_worker(self) -> None:
        from voicepad.tui.workers import warm_model

        result = warm_model(self.config)
        self.call_from_thread(self._on_model_ready, result)

    def _on_model_ready(self, result: ModelWarmResult) -> None:
        self._warm_result = result
        btn = self.query_one("#record-btn", Button)
        model_label = self.query_one("#header-model", Label)

        if result.error:
            self._set_status("error", f"model error: {result.error}")
            return

        fallback = "  cpu fallback" if result.fallback else ""
        model_label.update(
            f"[dim]{self.config.transcription_model}[/]  "
            f"[bold]{result.device}[/] [dim]{result.compute_type}{fallback}[/]"
        )
        self._set_status("ready", "ready")
        btn.label = "●  start recording"
        btn.disabled = False
        self._model_ready = True

    # ------------------------------------------------------------------
    # Record / stop
    # ------------------------------------------------------------------

    def action_toggle_recording(self) -> None:
        if not self._model_ready or self._transcribing:
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
        try:
            self._session.start()
        except AudioRecorderError as e:
            self._set_status("error", f"mic error: {e}")
            return

        self._recording = True
        self._record_start = time.monotonic()
        btn = self.query_one("#record-btn", Button)
        btn.label = "■  stop recording"
        btn.add_class("recording")
        self._set_status("recording", "recording…")
        self._start_timer()

    def _stop_recording(self) -> None:
        if self._session is None:
            return

        self._recording = False
        self._stop_timer()

        btn = self.query_one("#record-btn", Button)
        btn.label = "◌  transcribing…"
        btn.remove_class("recording")
        btn.add_class("transcribing")
        btn.disabled = True
        self._set_status("transcribing", "transcribing…")

        try:
            audio = self._session.stop()
        except AudioRecorderError as e:
            self._set_status("error", f"stop error: {e}")
            self._reset_button()
            return

        self._transcribing = True
        self._transcribe_worker(audio)

    # ------------------------------------------------------------------
    # Transcription
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

        if error:
            self._set_status("error", error)
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
                    md_path = self.config.markdown_path / f"{wav_path.stem}.md"
                    self.config.markdown_path.mkdir(parents=True, exist_ok=True)
                    md_path.write_text(_format_markdown(wav_path, result), encoding="utf-8")
                except Exception:
                    wav_path = None

        if result:
            # Update transcription panel
            tx_text = self.query_one("#tx-text", Label)
            tx_text.remove_class("placeholder")
            tx_text.update(result.text or "(no speech detected)")
            self.query_one("#tx-meta", Label).update(
                f"[dim]{result.duration_s:.1f}s  ·  {result.latency_ms:.0f}ms  ·  {result.device}[/]"
            )

            self._set_status("ready", "ready")

            # Add history entry
            entry = SessionEntry(
                index=len(self._entries),
                wav_path=wav_path,
                duration_s=result.duration_s,
                text=result.text,
                latency_ms=result.latency_ms,
                device=result.device,
            )
            self._entries.append(entry)
            self._add_history_entry(entry)

        self._reset_button()

    def _add_history_entry(self, entry: SessionEntry) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#empty-hint", Label).display = False

        lv = self.query_one("#history-list", ListView)
        name = entry.wav_path.stem if entry.wav_path else f"clip-{entry.index + 1}"
        preview = (entry.text[:70] + "…") if len(entry.text) > 70 else entry.text
        content = (
            f"[dim]{entry.timestamp}[/]  [bold]{name}[/]  "
            f"[dim]{entry.duration_s:.1f}s · {entry.latency_ms:.0f}ms[/]\n"
            f"  [dim italic]{preview or '(empty)'}[/]"
        )
        lv.append(ListItem(Label(content)))
        lv.scroll_end(animate=False)

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        self._timer_thread = None
        with contextlib.suppress(Exception):
            self.call_from_thread(self.query_one("#timer", Label).update, "")

    def _timer_loop(self) -> None:
        while self._recording:
            elapsed = time.monotonic() - self._record_start
            mins, secs = divmod(int(elapsed), 60)
            display = f"⏱  {mins:02d}:{secs:02d}" if mins else f"⏱  {elapsed:.1f}s"
            with contextlib.suppress(Exception):
                self.call_from_thread(self.query_one("#timer", Label).update, display)
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, state: str, message: str) -> None:
        dots = {"ready": "●", "recording": "◉", "transcribing": "◌", "error": "✕"}
        dot = dots.get(state, "○")
        label = self.query_one("#status", Label)
        label.remove_class("ready", "recording", "transcribing", "error")
        if state:
            label.add_class(state)
        label.update(f"{dot}  {message}")

    def _reset_button(self) -> None:
        btn = self.query_one("#record-btn", Button)
        btn.remove_class("recording", "transcribing")
        btn.label = "●  start recording"
        btn.disabled = not self._model_ready


# ---------------------------------------------------------------------------
# Markdown formatter
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
