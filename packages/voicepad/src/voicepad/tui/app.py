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
from textual.theme import BUILTIN_THEMES, Theme
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    MarkdownViewer,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option
from voicepad_core import (
    VALID_TRANSCRIPTION_MODELS,
    AudioRecorder,
    AudioRecorderError,
    ChunkResult,
    StreamingTranscriber,
    get_config,
)
from voicepad_core.config import Config
from voicepad_core.config.settings import get_config_with_metadata

from voicepad.tui.workers import ModelWarmResult, RecordingSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme — use a blue-tinted Catppuccin Mocha theme
# ---------------------------------------------------------------------------


_BASE_MOCHA = BUILTIN_THEMES["catppuccin-mocha"]
_BLUE_MOCHA = Theme(
    name="catppuccin-mocha-blue",
    primary="#89b4fa",  # Catppuccin Blue
    secondary="#74c7ec",  # Catppuccin Sapphire
    warning=_BASE_MOCHA.warning,
    error=_BASE_MOCHA.error,
    success=_BASE_MOCHA.success,
    accent="#89dceb",  # Catppuccin Sky
    foreground=_BASE_MOCHA.foreground,
    background=_BASE_MOCHA.background,
    surface=_BASE_MOCHA.surface,
    panel=_BASE_MOCHA.panel,
    boost=_BASE_MOCHA.boost,
    dark=_BASE_MOCHA.dark,
    luminosity_spread=_BASE_MOCHA.luminosity_spread,
    text_alpha=_BASE_MOCHA.text_alpha,
    variables={
        **_BASE_MOCHA.variables,
        "border": "#89b4fa",
    },
)

_THEME_NAME = _BLUE_MOCHA.name

_MD_PLACEHOLDER = """\
# voicepad

Select a recording from the list on the left to view its full transcription here.

Use the **⟳ retranscribe** button to re-run the model on the selected recording.
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
        self._streamer: StreamingTranscriber | None = None
        self._stream_chunks: list[ChunkResult] = []  # accumulated chunks in order
        self._entries: list[SessionEntry] = []
        self._record_start: float = 0.0
        self._timer_thread: threading.Thread | None = None
        self._warm_result: ModelWarmResult | None = None
        self._current_text: str = ""
        self._selected_entry_idx: int | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Static(id="header"):
            yield Label("voicepad", id="header-title")
            yield Label("󰔟  initialising", id="status")
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

            # ── Tab 3: Settings ────────────────────────────────────
            with TabPane("  settings  ", id="tab-settings"):
                with VerticalScroll(id="settings-scroll"):
                    yield Static(id="settings-fields")
                with Static(id="settings-footer"):
                    yield Label("", id="settings-status")
                    yield Button("󰆓  save", id="settings-save-btn")

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

        self.register_theme(_BLUE_MOCHA)
        self.theme = _THEME_NAME
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
        from utilityhub_config import get_config_path
        from voicepad_core.config import Config as _Config

        from voicepad.cli.config import _get_input_devices

        user_fields = {
            "recordings_path": "Where your WAV recordings are saved",
            "markdown_path": "Where your transcription files are saved",
            "transcription_model": "Whisper model to use for transcription",
            "input_device_index": "Microphone to record from",
        }

        # Build device options once — reused for the Select widget
        audio_devices = _get_input_devices()
        device_options: list[tuple[str, int]] = [("system default", -1)]
        device_options += [(f"[{d.index}]  {d.name}", d.index) for d in audio_devices]

        container = self.query_one("#settings-fields", Static)
        _, meta = get_config_with_metadata()

        # Show the config file path at the top so users know where to find it
        config_path = get_config_path("voicepad", format="yaml")
        path_label = Label(
            f"[dim]  {config_path}[/]",
            id="settings-config-path",
        )
        container.mount(path_label)

        for field_name, hint in user_fields.items():
            field_info = _Config.model_fields.get(field_name)
            if field_info is None:
                continue
            current_val = getattr(self.config, field_name)

            # Show field name only — no source label
            key_label = Label(
                f"[bold]{field_name}[/]",
                classes="settings-key",
            )
            hint_label = Label(f"[dim]{hint}[/]", classes="settings-hint")

            if field_name == "transcription_model":
                options = [(m, m) for m in VALID_TRANSCRIPTION_MODELS]
                current_str = str(current_val) if current_val is not None else "turbo"
                widget = Select(
                    options=options,
                    value=current_str if current_str in VALID_TRANSCRIPTION_MODELS else "turbo",
                    id="setting-transcription_model",
                    classes="settings-input",
                    allow_blank=False,
                )
            elif field_name == "input_device_index":
                # current_val is int | None; use -1 as the sentinel for "system default"
                current_idx = current_val if current_val is not None else -1
                valid_values = {v for _, v in device_options}
                widget = Select(
                    options=device_options,
                    value=current_idx if current_idx in valid_values else -1,
                    id="setting-input_device_index",
                    classes="settings-input",
                    allow_blank=False,
                )
            else:
                widget = Input(
                    value=str(current_val) if current_val is not None else "",
                    placeholder=str(field_info.default) if field_info.default is not None else "",
                    id=f"setting-{field_name}",
                    classes="settings-input",
                )

            row = Static(classes="settings-row")
            container.mount(row)
            row.mount(key_label, hint_label, widget)

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
                if field_name == "transcription_model":
                    sel = self.query_one("#setting-transcription_model", Select)
                    raw[field_name] = str(sel.value) if sel.value is not Select.BLANK else raw[field_name]
                elif field_name == "input_device_index":
                    sel = self.query_one("#setting-input_device_index", Select)
                    if sel.value is not Select.BLANK:
                        v = int(sel.value)  # ty:ignore[invalid-argument-type]
                        raw[field_name] = None if v == -1 else v
                else:
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
            # Always write to the global config — never a project-local file
            from utilityhub_config import get_config_path

            global_path = get_config_path("voicepad", format="yaml")
            write_config(new_config, "voicepad", path=global_path, format="yaml")

            model_changed = (
                new_config.transcription_model != self.config.transcription_model
                or new_config.transcription_device != self.config.transcription_device
                or new_config.transcription_compute_type != self.config.transcription_compute_type
            )
            object.__setattr__(self, "config", new_config)

            if model_changed and not self._recording and not self._transcribing:
                from voicepad_core.transcription import _model_cache

                _model_cache.clear()
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
        self._stream_chunks = []
        self._set_status("recording", "recording…")
        self._start_timer()

        # Start streaming transcriber — transcribes chunks during recording
        self._streamer = StreamingTranscriber(
            recorder=self._session._recorder,  # ty:ignore[invalid-argument-type]
            config=self.config,
            on_chunk=lambda chunk: self.call_from_thread(self._on_stream_chunk, chunk),
            on_error=lambda err: self.call_from_thread(self._set_status, "error", err),
        )
        self._streamer.start()

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
            if self._streamer:
                self._streamer._stop_event.set()
            return

        self._transcribing = True
        # Stop the streamer in a thread — it will transcribe the tail and call on_chunk(is_final=True)
        self._finalize_worker(audio)

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    @work(thread=True, name="finalize")
    def _finalize_worker(self, audio: np.ndarray) -> None:
        """Stop the streamer (transcribes tail) then save the full recording."""
        if self._streamer:
            self._streamer.stop()  # blocks until final chunk callback fires
        self.call_from_thread(self._save_recording, audio)

    def _on_stream_chunk(self, chunk: ChunkResult) -> None:
        """Called from the streaming thread for each transcribed chunk."""
        if chunk.text:
            self._stream_chunks.append(chunk)

        # Update transcription panel with all accumulated text so far
        full_text = " ".join(c.text for c in self._stream_chunks).strip()
        if full_text:
            tx_text = self.query_one("#tx-text", Label)
            tx_text.remove_class("placeholder")
            tx_text.update(full_text)
            self.query_one("#transcription", Static).scroll_end(animate=False)

        if chunk.is_final:
            self._transcribing = False
            elapsed = time.monotonic() - self._record_start
            self.query_one("#tx-meta", Label).update(f"[dim]{elapsed:.1f}s  ·  streaming[/]")
            self._set_status("ready", "ready")

    def _save_recording(self, audio: np.ndarray) -> None:
        """Save WAV + markdown and add history entry after streaming completes."""
        full_text = " ".join(c.text for c in self._stream_chunks).strip()
        if not full_text:
            return

        self._current_text = full_text
        self.query_one("#tx-copy-btn", Button).disabled = False

        # Save WAV
        wav_path: Path | None = None
        md_path: Path | None = None
        recorder_ref: AudioRecorder | None = self._session._recorder if self._session else None
        if recorder_ref is not None:
            wav_path = recorder_ref.make_wav_path()
            try:
                recorder_ref.save_wav(audio, wav_path)
                # Build a synthetic TranscriptionResult-like object for _format_markdown
                md_path = self.config.markdown_path / f"{wav_path.stem}.md"
                self.config.markdown_path.mkdir(parents=True, exist_ok=True)
                duration_s = len(audio) / 16000
                md_path.write_text(
                    _format_markdown_streaming(wav_path, full_text, duration_s, self._stream_chunks),
                    encoding="utf-8",
                )
            except Exception:
                wav_path = None
                md_path = None

        entry = SessionEntry(
            index=len(self._entries),
            wav_path=wav_path,
            md_path=md_path,
            duration_s=len(audio) / 16000,
            text=full_text,
            latency_ms=0.0,
            device=self._stream_chunks[-1].device if self._stream_chunks else "cuda",
        )
        self._entries.append(entry)
        self._add_history_entry(entry)

    # ------------------------------------------------------------------
    # Live transcription (record tab)
    # ------------------------------------------------------------------

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
            idx = int(event.option.id)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
        label.update(f"󰑊  recording…  󰔛 {timer_str}")

    def _refresh_status_label(self) -> None:
        label = self.query_one("#status", Label)
        if "󰔛" in str(label.renderable):  # ty:ignore[unresolved-attribute]
            label.update("󰔟  transcribing…")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, state: str, message: str) -> None:
        dots = {"ready": "", "recording": "󰑊", "transcribing": "󰔟", "error": "󰅙"}
        dot = dots.get(state, "󰔟")
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


def _format_markdown_streaming(
    wav_path: Path,
    text: str,
    duration_s: float,
    chunks: list[ChunkResult],
) -> str:
    """Format a markdown file for a streaming transcription."""
    latest_chunk = next((chunk for chunk in reversed(chunks) if chunk.text), None)
    device = latest_chunk.device if latest_chunk else "unknown"
    language = latest_chunk.language if latest_chunk else "en"
    language_probability = latest_chunk.language_probability if latest_chunk else 0.0
    latency_ms = sum(chunk.latency_ms for chunk in chunks)

    lines = [
        "# Transcription",
        "",
        f"**File:** `{wav_path.name}`",
        f"**Model:** {device} / live",
        f"**Language:** {language} ({language_probability * 100:.1f}%)",
        f"**Duration:** {duration_s:.1f}s",
        f"**Latency:** {latency_ms:.0f}ms",
        "**Mode:** streaming",
        "",
        "---",
        "",
        "## Text",
        "",
        text or "*(no speech detected)*",
        "",
    ]

    segments = [segment for chunk in chunks for segment in chunk.segments if segment.text]
    if segments:
        lines += ["## Segments", ""]
        for segment in segments:
            lines.append(f"- `[{segment.start:.1f}s → {segment.end:.1f}s]` {segment.text}")

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
