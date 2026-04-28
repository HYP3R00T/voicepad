"""VoicePad Textual TUI application."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    MarkdownViewer,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option
from voicepad_core import AudioRecorder, AudioRecorderError, get_config
from voicepad_core.config import Config
from voicepad_core.config.settings import get_config_with_metadata

from voicepad.tui.workers import ModelWarmResult, RecordingSession, TranscriptionJob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme
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

_MD_PLACEHOLDER = """\
# voicepad

Select a recording from the list on the left to view its full transcription here.

Use the **⟳ retranscribe** button to re-run the model on the selected recording.
"""

_TRANSCRIBE_PLACEHOLDER = """\
# transcribe a file

Enter the path to any audio file above and press **Transcribe** (or hit Enter).

Supported formats: WAV, MP3, FLAC, OGG, M4A, and anything else soundfile can read.
"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class SessionEntry:
    index: int
    wav_path: Path | None
    md_path: Path | None
    duration_s: float
    text: str
    latency_ms: float
    device: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class VoicePadApp(App[None]):
    """VoicePad — local dictation with Whisper."""

    TITLE = "VoicePad"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("space", "toggle_recording", "Record / Stop", show=True),
        Binding("c", "copy_transcription", "Copy", show=True),
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
        self._current_text: str = ""
        # history tab: currently selected entry index
        self._selected_entry_idx: int | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Static(id="header"):
            yield Label("voicepad", id="header-title")
            yield Label("○  initialising", id="status")
            yield Label("loading…", id="header-model")

        with Static(id="body"), TabbedContent(id="tabs"):
            # ── Tab 1: Record ──────────────────────────────────────
            with TabPane("  record  ", id="tab-record"):
                tx = Static(id="transcription")
                tx.border_title = "transcription"
                yield tx

            # ── Tab 2: History + Retranscribe ──────────────────────
            with TabPane("  history  ", id="tab-history"), Static(id="history-section"):
                hist_list = Static(id="history-list-pane")
                hist_list.border_title = "recordings"
                yield hist_list

                with Static(id="history-view-pane"):
                    yield MarkdownViewer(
                        _MD_PLACEHOLDER,
                        id="history-viewer",
                        show_table_of_contents=False,
                    )
                    yield Button(
                        "⟳  retranscribe",
                        id="retranscribe-btn",
                        disabled=True,
                    )

            # ── Tab 3: Transcribe any file ─────────────────────────
            with TabPane("  transcribe file  ", id="tab-transcribe"):
                with Static(id="tf-section"):
                    yield Input(
                        placeholder="path to audio file…",
                        id="tf-input",
                    )
                    yield Button("▶  transcribe", id="tf-btn", disabled=True)
                yield MarkdownViewer(
                    _TRANSCRIBE_PLACEHOLDER,
                    id="tf-viewer",
                    show_table_of_contents=False,
                )

            # ── Tab 4: Settings ────────────────────────────────────
            with TabPane("  settings  ", id="tab-settings"):
                with VerticalScroll(id="settings-scroll"):
                    yield Static(id="settings-fields")
                with Static(id="settings-footer"):
                    yield Label("", id="settings-status")
                    yield Button("💾  save", id="settings-save-btn")

        yield Footer()

    def on_mount(self) -> None:
        # ── Tab 1: record ──
        tx = self.query_one("#transcription", Static)
        tx.mount(Label("speak and press space to begin…", id="tx-text", classes="placeholder"))
        tx.mount(Label("", id="tx-meta"))
        tx.mount(Button("⎘  copy", id="tx-copy-btn", disabled=True))

        # ── Tab 2: history ──
        hist_list = self.query_one("#history-list-pane", Static)
        hist_list.mount(OptionList(id="history-options"))

        self.register_theme(VOICEPAD_THEME)
        self.theme = "voicepad"
        self._load_history_from_disk()
        self._populate_settings()
        self._warm_model_worker()

    # ------------------------------------------------------------------
    # History — pre-populate from disk
    # ------------------------------------------------------------------

    def _load_history_from_disk(self) -> None:
        md_dir = self.config.markdown_path
        if not md_dir.exists():
            return
        for md_path in sorted(md_dir.glob("*.md")):
            entry = _parse_markdown_entry(
                md_path, index=len(self._entries), recordings_path=self.config.recordings_path
            )
            if entry is not None:
                self._entries.append(entry)
                self._add_history_entry(entry)

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------

    def _populate_settings(self) -> None:
        """Build the settings form — only user-facing fields shown."""
        from voicepad_core.config import Config as _Config

        # Fields shown in the TUI — technical knobs stay in voicepad.yaml only
        user_fields = {
            "recordings_path": "Where your WAV recordings are saved",
            "markdown_path": "Where your transcription files are saved",
            "transcription_model": (
                "Whisper model to use. Options: tiny, base, small, medium, "
                "large-v3, turbo (recommended), large-v3-turbo"
            ),
            "input_device_index": "Microphone device index. Leave blank for system default",
        }

        container = self.query_one("#settings-fields", Static)
        _, meta = get_config_with_metadata()

        for field_name, hint in user_fields.items():
            field_info = _Config.model_fields.get(field_name)
            if field_info is None:
                continue
            current_val = getattr(self.config, field_name)
            src = meta.get_source(field_name)
            source_label = src.source if src else "default"

            key_label = Label(
                f"[bold]{field_name}[/]  [dim]({source_label})[/]",
                classes="settings-key",
            )
            hint_label = Label(f"[dim]{hint}[/]", classes="settings-hint")
            inp = Input(
                value=str(current_val) if current_val is not None else "",
                placeholder=str(field_info.default) if field_info.default is not None else "",
                id=f"setting-{field_name}",
                classes="settings-input",
            )

            row = Static(classes="settings-row")
            container.mount(row)
            row.mount(key_label, hint_label, inp)

    @on(Button.Pressed, "#settings-save-btn")
    def on_settings_save(self) -> None:
        """Read user-facing inputs, merge with existing config, write to voicepad.yaml."""
        from utilityhub_config import write_config
        from voicepad_core.config import Config as _Config

        user_fields = ["recordings_path", "markdown_path", "transcription_model", "input_device_index"]

        status = self.query_one("#settings-status", Label)
        errors: list[str] = []

        # Start from current config values (preserves hidden fields)
        raw = self.config.model_dump(mode="json")

        for field_name in user_fields:
            field_info = _Config.model_fields.get(field_name)
            if field_info is None:
                continue
            with contextlib.suppress(Exception):
                inp = self.query_one(f"#setting-{field_name}", Input)
                val_str = inp.value.strip()
                annotation = field_info.annotation
                if val_str == "" or val_str.lower() == "none":
                    raw[field_name] = None
                elif annotation in (int, "int") or "int" in str(annotation):
                    try:
                        raw[field_name] = int(val_str)
                    except ValueError:
                        errors.append(f"{field_name}: expected a number")
                else:
                    raw[field_name] = val_str

        if errors:
            status.update(f"[red]✕  {'; '.join(errors)}[/]")
            return

        try:
            new_config = _Config(**raw)
            write_config(new_config, "voicepad", path=Path("voicepad.yaml"), format="yaml")

            model_changed = (
                new_config.transcription_model != self.config.transcription_model
                or new_config.transcription_device != self.config.transcription_device
                or new_config.transcription_compute_type != self.config.transcription_compute_type
            )
            object.__setattr__(self, "config", new_config)

            if model_changed and not self._recording and not self._transcribing:
                from voicepad_core.transcription import _batched_cache, _model_cache

                _model_cache.clear()
                _batched_cache.clear()
                self._model_ready = False
                self._set_status("transcribing", "loading model…")
                self.query_one("#header-model", Label).update("loading…")
                self._warm_model_worker()
                status.update("[green]✓  saved — reloading model[/]")
            else:
                status.update("[green]✓  saved[/]")

            self.set_timer(3.0, lambda: status.update(""))
        except Exception as e:
            status.update(f"[red]✕  {e}[/]")

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
        self._model_ready = True
        # Enable tf-btn if there's already a path in the input
        with contextlib.suppress(Exception):
            val = self.query_one("#tf-input", Input).value.strip()
            self.query_one("#tf-btn", Button).disabled = not bool(val)

    # ------------------------------------------------------------------
    # Record / stop
    # ------------------------------------------------------------------

    def action_toggle_recording(self) -> None:
        active = self.query_one("#tabs", TabbedContent).active
        if active != "tab-record":
            return
        if not self._model_ready or self._transcribing:
            return
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self._session = RecordingSession(config=self.config)
        try:
            self._session.start()
        except AudioRecorderError as e:
            self._set_status("error", f"mic error: {e}")
            return

        self._recording = True
        self._record_start = time.monotonic()
        self._set_status("recording", "recording…")
        self._start_timer()

    def _stop_recording(self) -> None:
        if self._session is None:
            return

        self._recording = False
        self._stop_timer()
        self._set_status("transcribing", "transcribing…")

        try:
            audio = self._session.stop()
        except AudioRecorderError as e:
            self._set_status("error", f"stop error: {e}")
            return

        self._transcribing = True
        self._transcribe_worker(audio)

    # ------------------------------------------------------------------
    # Live transcription (record tab)
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
            return

        wav_path: Path | None = None
        md_path: Path | None = None
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
                    md_path = None

        if result:
            tx_text = self.query_one("#tx-text", Label)
            tx_text.remove_class("placeholder")
            tx_text.update(result.text or "(no speech detected)")
            self.query_one("#tx-meta", Label).update(
                f"[dim]{result.duration_s:.1f}s  ·  {result.latency_ms:.0f}ms  ·  {result.device}[/]"
            )
            self.query_one("#transcription", Static).scroll_end(animate=False)
            self._set_status("ready", "ready")

            self._current_text = result.text or ""
            self.query_one("#tx-copy-btn", Button).disabled = not bool(self._current_text)

            entry = SessionEntry(
                index=len(self._entries),
                wav_path=wav_path,
                md_path=md_path,
                duration_s=result.duration_s,
                text=result.text,
                latency_ms=result.latency_ms,
                device=result.device,
            )
            self._entries.append(entry)
            self._add_history_entry(entry)

    def _add_history_entry(self, entry: SessionEntry) -> None:
        ol = self.query_one("#history-options", OptionList)
        name = entry.wav_path.stem if entry.wav_path else f"clip-{entry.index + 1}"
        label = (
            f"[bold]{entry.timestamp}[/]  [dim]{name}[/]\n"
            f"  [dim]{entry.duration_s:.1f}s · {entry.latency_ms:.0f}ms · {entry.device}[/]"
        )
        ol.add_option(Option(label, id=str(entry.index)))
        ol.highlighted = ol.option_count - 1

    # ------------------------------------------------------------------
    # History tab — select entry → show markdown + enable retranscribe
    # ------------------------------------------------------------------

    @on(OptionList.OptionSelected, "#history-options")
    def on_history_option_selected(self, event: OptionList.OptionSelected) -> None:
        with contextlib.suppress(Exception):
            idx = int(event.option.id)  # type: ignore[arg-type]
            if 0 <= idx < len(self._entries):
                self._selected_entry_idx = idx
                entry = self._entries[idx]
                self.query_one("#retranscribe-btn", Button).disabled = not self._model_ready or entry.wav_path is None
                if entry.md_path and entry.md_path.exists():
                    self._load_history_viewer(entry.md_path)

    @work(name="md-view")
    async def _load_history_viewer(self, md_path: Path) -> None:
        viewer = self.query_one("#history-viewer", MarkdownViewer)
        await viewer.go(md_path.resolve())

    # ------------------------------------------------------------------
    # Retranscribe (history tab)
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#retranscribe-btn")
    def on_retranscribe_btn_pressed(self) -> None:
        if self._selected_entry_idx is None or not self._model_ready:
            return
        entry = self._entries[self._selected_entry_idx]
        if entry.wav_path and entry.wav_path.exists():
            self.query_one("#retranscribe-btn", Button).disabled = True
            self._retranscribe_file(entry.wav_path, entry.md_path)

    @work(thread=True, name="retranscribe")
    def _retranscribe_file(self, wav_path: Path, md_path: Path | None) -> None:
        from voicepad_core.transcription import transcribe_buffer

        self.call_from_thread(self._set_status, "transcribing", f"retranscribing {wav_path.name}…")
        try:
            audio, _sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            result = transcribe_buffer(audio, self.config)
            error: str | None = None
        except Exception as e:
            result = None
            error = str(e)

        self.call_from_thread(self._on_retranscribe_done, wav_path, md_path, result, error)

    def _on_retranscribe_done(self, wav_path: Path, md_path: Path | None, result, error: str | None) -> None:
        self.query_one("#retranscribe-btn", Button).disabled = False
        if error:
            self._set_status("error", error)
            return

        if result:
            # Overwrite markdown and update the entry
            out_md = md_path or (self.config.markdown_path / f"{wav_path.stem}.md")
            self.config.markdown_path.mkdir(parents=True, exist_ok=True)
            out_md.write_text(_format_markdown(wav_path, result), encoding="utf-8")
            self._set_status("ready", "ready")

            # Update the in-memory entry if it exists
            if self._selected_entry_idx is not None:
                entry = self._entries[self._selected_entry_idx]
                self._entries[self._selected_entry_idx] = SessionEntry(
                    index=entry.index,
                    wav_path=entry.wav_path,
                    md_path=out_md,
                    duration_s=result.duration_s,
                    text=result.text,
                    latency_ms=result.latency_ms,
                    device=result.device,
                    timestamp=entry.timestamp,
                )

            # Reload the viewer with the fresh markdown
            self._load_history_viewer(out_md)

    # ------------------------------------------------------------------
    # Transcribe file tab
    # ------------------------------------------------------------------

    @on(Input.Changed, "#tf-input")
    def on_tf_input_changed(self, event: Input.Changed) -> None:
        self.query_one("#tf-btn", Button).disabled = not self._model_ready or not event.value.strip()

    @on(Input.Submitted, "#tf-input")
    def on_tf_input_submitted(self, _event: Input.Submitted) -> None:
        self._start_tf_transcription()

    @on(Button.Pressed, "#tf-btn")
    def on_tf_btn_pressed(self) -> None:
        self._start_tf_transcription()

    def _start_tf_transcription(self) -> None:
        path_str = self.query_one("#tf-input", Input).value.strip()
        if not path_str or not self._model_ready:
            return
        audio_path = Path(path_str)
        if not audio_path.exists():
            self._set_status("error", f"file not found: {audio_path.name}")
            return
        self.query_one("#tf-btn", Button).disabled = True
        self._transcribe_file_worker(audio_path)

    @work(thread=True, name="tf-transcribe")
    def _transcribe_file_worker(self, audio_path: Path) -> None:
        from voicepad_core.transcription import transcribe_buffer

        self.call_from_thread(self._set_status, "transcribing", f"transcribing {audio_path.name}…")
        try:
            audio, _sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            result = transcribe_buffer(audio, self.config)
            error: str | None = None
        except Exception as e:
            result = None
            error = str(e)

        self.call_from_thread(self._on_tf_done, audio_path, result, error)

    def _on_tf_done(self, audio_path: Path, result, error: str | None) -> None:
        self.query_one("#tf-btn", Button).disabled = False
        if error:
            self._set_status("error", error)
            return

        if result:
            # Save markdown alongside the audio file (or in markdown_path if read-only)
            md_path = self.config.markdown_path / f"{audio_path.stem}.md"
            self.config.markdown_path.mkdir(parents=True, exist_ok=True)
            md_path.write_text(_format_markdown(audio_path, result), encoding="utf-8")
            self._set_status("ready", f"done — {audio_path.name}")
            self._load_tf_viewer(md_path)

    @work(name="tf-view")
    async def _load_tf_viewer(self, md_path: Path) -> None:
        viewer = self.query_one("#tf-viewer", MarkdownViewer)
        await viewer.go(md_path.resolve())

    # ------------------------------------------------------------------
    # Copy transcription (record tab)
    # ------------------------------------------------------------------

    def action_copy_transcription(self) -> None:
        if not self._current_text:
            return
        _copy_to_clipboard(self._current_text)
        with contextlib.suppress(Exception):
            btn = self.query_one("#tx-copy-btn", Button)
            btn.label = "✓  copied"
            self.set_timer(1.5, lambda: setattr(btn, "label", "⎘  copy"))

    @on(Button.Pressed, "#tx-copy-btn")
    def on_copy_btn_pressed(self) -> None:
        self.action_copy_transcription()

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        self._timer_thread = None
        with contextlib.suppress(Exception):
            self.call_from_thread(self._refresh_status_label)

    def _timer_loop(self) -> None:
        while self._recording:
            elapsed = time.monotonic() - self._record_start
            mins, secs = divmod(int(elapsed), 60)
            timer_str = f"{mins:02d}:{secs:02d}" if mins else f"{elapsed:.1f}s"
            with contextlib.suppress(Exception):
                self.call_from_thread(self._update_status_with_timer, timer_str)
            time.sleep(0.1)

    def _update_status_with_timer(self, timer_str: str) -> None:
        label = self.query_one("#status", Label)
        label.update(f"◉  recording…  ⏱ {timer_str}")

    def _refresh_status_label(self) -> None:
        label = self.query_one("#status", Label)
        if "⏱" in str(label.renderable):
            label.update("◌  transcribing…")

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


# ---------------------------------------------------------------------------
# Markdown formatter / parser
# ---------------------------------------------------------------------------


def _format_markdown(audio_path: Path, result) -> str:
    lines = [
        "# Transcription",
        "",
        f"**File:** `{audio_path.name}`",
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


def _parse_markdown_entry(md_path: Path, index: int, recordings_path: Path | None = None) -> SessionEntry | None:
    """Parse a transcription markdown file back into a SessionEntry for history display.

    recordings_path: where to look for the corresponding WAV file.
    Falls back to a sibling 'recordings' directory if not provided.
    """
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception:
        return None

    duration_s = 0.0
    latency_ms = 0.0
    device = "unknown"
    wav_name: str | None = None

    for line in content.splitlines():
        if line.startswith("**File:**"):
            wav_name = line.split("`")[1] if "`" in line else None
        elif line.startswith("**Duration:**") or line.startswith("**Audio duration:**"):
            with contextlib.suppress(Exception):
                duration_s = float(line.split(":**")[1].strip().rstrip("s"))
        elif line.startswith("**Latency:**") or line.startswith("**Transcription latency:**"):
            with contextlib.suppress(Exception):
                latency_ms = float(line.split(":**")[1].strip().rstrip("ms"))
        elif line.startswith("**Model:**"):
            with contextlib.suppress(Exception):
                device = line.split("**Model:**")[1].strip().split("/")[0].strip()

    in_text_section = False
    text_lines: list[str] = []
    for line in content.splitlines():
        if line.strip() == "## Text":
            in_text_section = True
            continue
        if in_text_section:
            if line.startswith("## "):
                break
            if line.strip() and not line.startswith("---"):
                text_lines.append(line)
    text = " ".join(text_lines).strip()

    if not text or text == "*(no speech detected)*":
        return None

    # Resolve WAV path: check configured recordings_path first, then sibling directory
    wav_path: Path | None = None
    if wav_name:
        candidates: list[Path] = []
        if recordings_path is not None:
            candidates.append(recordings_path / wav_name)
        candidates.append(md_path.parent.parent / "recordings" / wav_name)
        for candidate in candidates:
            if candidate.exists():
                wav_path = candidate
                break

    timestamp = ""
    stem = md_path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        date_part = parts[-2]
        time_part = parts[-1]
        with contextlib.suppress(Exception):
            timestamp = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[0:2]}:{time_part[2:4]}"

    return SessionEntry(
        index=index,
        wav_path=wav_path,
        md_path=md_path,
        duration_s=duration_s,
        text=text,
        latency_ms=latency_ms,
        device=device,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Clipboard helper
# ---------------------------------------------------------------------------


def _copy_to_clipboard(text: str) -> None:
    """Copy text to the system clipboard using pyperclip (cross-platform)."""
    try:
        import pyperclip

        pyperclip.copy(text)
    except Exception as e:
        logger.warning(f"Clipboard copy failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the VoicePad TUI."""
    config = get_config()
    app = VoicePadApp(config)
    app.run()
