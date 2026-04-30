"""VoicePad Textual TUI application."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import soundfile as sf
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    Link,
    Markdown,
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
# Version — read dynamically from package metadata
# ---------------------------------------------------------------------------

try:
    _APP_VERSION = f"v{_pkg_version('voicepad')}"
except Exception:
    _APP_VERSION = "dev"

# ---------------------------------------------------------------------------
# Theme — Catppuccin Mocha with blue primary instead of pink
# ---------------------------------------------------------------------------

_THEME_NAME = "catppuccin-mocha-blue"

_CATPPUCCIN_MOCHA_BLUE = Theme(
    name=_THEME_NAME,
    primary="#89b4fa",  # Catppuccin Mocha Blue
    secondary="#74c7ec",  # Catppuccin Mocha Sapphire
    warning="#FAE3B0",
    error="#F28FAD",
    success="#ABE9B3",
    accent="#fab387",
    foreground="#cdd6f4",
    background="#181825",
    surface="#313244",
    panel="#45475a",
    variables={
        "input-cursor-foreground": "#11111b",
        "input-cursor-background": "#f5e0dc",
        "input-selection-background": "#9399b2 30%",
        "border": "#89b4fa",
        "border-blurred": "#585b70",
        "footer-background": "#45475a",
        "footer-key-foreground": "#89b4fa",
        "block-cursor-foreground": "#1e1e2e",
        "block-cursor-text-style": "none",
        "button-color-foreground": "#181825",
    },
)


_MD_PLACEHOLDER = """\
# voicepad

Select a recording from the list on the left to view its full transcription here.

Use the **⟳ retranscribe** button to re-run the model on the selected recording.
"""

# ---------------------------------------------------------------------------
# Info Modal Screen
# ---------------------------------------------------------------------------


class InfoModal(ModalScreen[None]):
    """Modal screen showing app info, version, and sponsor information."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("i", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Static(id="info-dialog"):
            # ── 1. Header ─────────────────────────────────────────
            yield Static("󰍬  VoicePad", id="info-title")

            # ── 2. Tagline ────────────────────────────────────────
            yield Static(
                "Your voice, your data.\nTranscription that never leaves your machine.",
                id="info-subtitle",
            )

            # ── 3. Features box ───────────────────────────────────
            with Static(id="info-guarantees"):
                yield Static("󰒍  Fully local processing", classes="guarantee-line")
                yield Static("󰌨  GPU-accelerated transcription", classes="guarantee-line")
                yield Static("󰍹  No cloud. No tracking. No data leaks.", classes="guarantee-line")

            # ── 4. Philosophy ─────────────────────────────────────
            yield Static(
                "Built with 󰋑 using Python, Textual, and Whisper",
                id="info-philosophy",
            )

            # ── 5. Separator ──────────────────────────────────────
            yield Static("", id="info-divider")

            # ── 6. CTA ────────────────────────────────────────────
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

            # ── 7. Micro text ─────────────────────────────────────
            yield Static(
                "Privacy-first tools grow through community support.",
                id="info-microcopy",
            )

            # ── 8. Metadata ───────────────────────────────────────
            yield Static(
                f"{_APP_VERSION}  •  Rajesh Das (HYP3R00T)  •  MIT License",
                id="info-meta",
            )

            # ── 9. Separator ──────────────────────────────────────
            yield Static("", id="info-divider2")

            # ── 10. Close ─────────────────────────────────────────
            yield Button("Close", variant="default", id="info-close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close the modal when button is pressed."""
        self.dismiss()


# ---------------------------------------------------------------------------
# First-run / Setup Modal
# ---------------------------------------------------------------------------


class SetupModal(ModalScreen[str]):
    """Shown on first run or when the selected model is not downloaded.

    Lets the user confirm (or change) the model, then downloads it with a
    progress indicator. Dismisses with the chosen model name when done.
    """

    BINDINGS: list[Binding] = []  # no escape — must complete setup

    def __init__(self, config: Config, config_missing: bool) -> None:
        super().__init__()
        self._config = config
        self._config_missing = config_missing
        self._downloading = False
        self._pending_model: str = config.transcription_model
        self._timer: object | None = None
        self._dot_count: int = 0
        self._chosen_model: str = config.transcription_model

    def compose(self) -> ComposeResult:
        with Static(id="setup-dialog"):
            yield Static("󰍬  Welcome to VoicePad", id="setup-title")

            if self._config_missing:
                yield Static(
                    "No config file found — using defaults.\nYou can change settings later from the Settings tab.",
                    id="setup-subtitle",
                )
            else:
                yield Static(
                    "The selected model is not downloaded yet.\nChoose a model and VoicePad will download it now.",
                    id="setup-subtitle",
                )

            yield Static("Select a Whisper model:", id="setup-model-label")
            yield Select(
                options=[(m, m) for m in VALID_TRANSCRIPTION_MODELS],
                value=self._config.transcription_model,
                id="setup-model-select",
                allow_blank=False,
            )

            yield Static("", id="setup-model-hint")
            yield Static("", id="setup-status")
            yield Button("Download & Start", id="setup-start-btn", variant="primary")

    def on_mount(self) -> None:
        self._update_hint(self._config.transcription_model)
        self._timer: object | None = None

    @on(Select.Changed, "#setup-model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._update_hint(str(event.value))

    def _update_hint(self, model: str) -> None:
        _hints: dict[str, str] = {
            "tiny": "~40 MB · fastest · low accuracy · CPU",
            "tiny.en": "~40 MB · fastest · English only · CPU",
            "base": "~75 MB · very fast · fair accuracy · CPU",
            "base.en": "~75 MB · very fast · English only · CPU",
            "small": "~250 MB · fast · good accuracy · ~1 GB VRAM",
            "small.en": "~250 MB · fast · English only · ~1 GB VRAM",
            "medium": "~770 MB · moderate · very good · ~2 GB VRAM",
            "medium.en": "~770 MB · moderate · English only · ~2 GB VRAM",
            "large-v1": "~1.5 GB · slow · excellent · ~5 GB VRAM",
            "large-v2": "~1.5 GB · slow · excellent · ~5 GB VRAM",
            "large-v3": "~1.5 GB · slow · best accuracy · ~5 GB VRAM",
            "large": "~1.5 GB · slow · excellent · ~5 GB VRAM",
            "large-v3-turbo": "~800 MB · recommended · excellent · ~3 GB VRAM",
            "turbo": "~800 MB · recommended · excellent · ~3 GB VRAM",
            "distil-small.en": "~135 MB · fast · English only · ~1 GB VRAM",
            "distil-medium.en": "~395 MB · moderate · English only · ~1 GB VRAM",
            "distil-large-v2": "~760 MB · fast · English only · ~3 GB VRAM",
            "distil-large-v3": "~760 MB · fast · English only · ~3 GB VRAM",
            "distil-large-v3.5": "~760 MB · fast · English only · ~3 GB VRAM",
        }
        hint = _hints.get(model, "")
        self.query_one("#setup-model-hint", Static).update(f"[dim]{hint}[/]" if hint else "")

    @on(Button.Pressed, "#setup-start-btn")
    def on_start_pressed(self) -> None:
        if self._downloading:
            return
        sel = self.query_one("#setup-model-select", Select)
        if sel.value is Select.BLANK:
            return
        model = str(sel.value)
        self._downloading = True
        self._pending_model = model
        self.query_one("#setup-start-btn", Button).disabled = True
        self._download_model(model)

    @work(thread=True, name="setup-download")
    def _download_model(self, model: str) -> None:
        from voicepad_core import ensure_model_downloaded, model_downloaded
        from voicepad_core.transcription import TranscriptionError

        self.app.call_from_thread(self._set_setup_status, f"Checking '{model}'…")

        if model_downloaded(model, self._config):
            self.app.call_from_thread(self._on_download_done, model, None)
            return

        self.app.call_from_thread(
            self._set_setup_status,
            f"Downloading '{model}'… this may take a few minutes.",
        )
        self.app.call_from_thread(self._show_progress)

        try:
            ensure_model_downloaded(model, self._config)
            self.app.call_from_thread(self._on_download_done, model, None)
        except TranscriptionError as e:
            self.app.call_from_thread(self._on_download_done, model, str(e))

    def _show_progress(self) -> None:
        self._dot_count = 0

        def _tick() -> None:
            self._dot_count = (self._dot_count + 1) % 4
            dots = "." * self._dot_count
            self._set_setup_status(f"Downloading '{self._pending_model}'{dots}")

        self._timer = self.set_interval(0.5, _tick)

    def _set_setup_status(self, msg: str) -> None:
        self.query_one("#setup-status", Static).update(msg)

    def _on_download_done(self, model: str, error: str | None) -> None:
        if self._timer is not None:
            self._timer.stop()  # type: ignore[union-attr]
            self._timer = None
        if error:
            self._set_setup_status(f"[red]Download failed: {error}[/]")
            self.query_one("#setup-start-btn", Button).disabled = False
            self._downloading = False
        else:
            self._chosen_model = model
            self._set_setup_status(f"[green]✓  '{model}' ready — starting…[/]")
            self.app.call_after_refresh(self._do_dismiss)

    def _do_dismiss(self) -> None:
        self.dismiss(self._chosen_model)


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
        Binding("i", "show_info", "Info", show=True, key_display="i"),
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
            yield Label("VoicePad", id="header-title")
            yield Label(_APP_VERSION, id="header-version")
            yield Label("󰔟  initialising", id="status")
            yield Label("loading…", id="header-model")

        yield Static(id="header-rule")

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
                        open_links=False,
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

        self.register_theme(_CATPPUCCIN_MOCHA_BLUE)
        self.theme = _THEME_NAME
        self._load_history_from_disk()
        self._populate_settings()
        self._check_first_run()

    # ------------------------------------------------------------------
    # First-run check
    # ------------------------------------------------------------------

    def _check_first_run(self) -> None:
        """Show setup modal if config is missing or model not downloaded."""
        from utilityhub_config import get_config_path
        from voicepad_core import model_downloaded

        config_path = get_config_path("voicepad", format="yaml")
        config_missing = not config_path.exists()
        model_ready = model_downloaded(self.config.transcription_model, self.config)

        if config_missing or not model_ready:
            self.push_screen(
                SetupModal(self.config, config_missing=config_missing),
                callback=self._on_setup_done,
            )
        else:
            self._warm_model_worker()

    def _on_setup_done(self, chosen_model: str) -> None:
        """Called when the setup modal dismisses with the chosen model name.

        Always writes the config file — this is the first-run setup, so we
        want to persist defaults + chosen model regardless of what was selected.
        """
        from utilityhub_config import get_config_path, write_config
        from voicepad_core.config import Config as _Config

        raw = self.config.model_dump(mode="json")
        raw["transcription_model"] = chosen_model

        try:
            new_config = _Config(**raw)
            global_path = get_config_path("voicepad", format="yaml")
            global_path.parent.mkdir(parents=True, exist_ok=True)
            write_config(new_config, "voicepad", path=global_path, format="yaml")
            object.__setattr__(self, "config", new_config)
            logger.info(f"Config written to {global_path}")
        except Exception as e:
            logger.warning(f"Could not write config: {e}")

        self._refresh_config_path_label()
        self._warm_model_worker()

    def _refresh_config_path_label(self) -> None:
        """Update the settings config path label to reflect current file state."""
        from utilityhub_config import get_config_path

        with contextlib.suppress(Exception):
            config_path = get_config_path("voicepad", format="yaml")
            exists_hint = "" if config_path.exists() else "  [dim red](not yet created)[/]"
            self.query_one("#settings-config-path", Label).update(f"[dim]config file:[/]  {config_path}{exists_hint}")

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

        # Show the config file path at the top — indicate if it exists or not
        config_path = get_config_path("voicepad", format="yaml")
        exists_hint = "" if config_path.exists() else "  [dim red](not yet created)[/]"
        path_label = Label(
            f"[dim]config file:[/]  {config_path}{exists_hint}",
            id="settings-config-path",
        )
        container.mount(path_label)

        for field_name, hint in user_fields.items():
            field_info = _Config.model_fields.get(field_name)
            if field_info is None:
                continue
            current_val = getattr(self.config, field_name)

            # Single-line label: "field_name  —  hint description"
            key_label = Label(
                f"[bold]{field_name}[/]  [dim]—  {hint}[/]",
                classes="settings-key",
            )

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
            row.mount(key_label, widget)

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
                        v = int(sel.value)
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
                self.query_one("#header-model", Label).update("[dim]M:[/] loading…")
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
            f"[dim]model:[/] {self.config.transcription_model}  [dim]device:[/] {result.device}{fallback}"
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
            recorder=self._session._recorder,
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

    @on(Markdown.LinkClicked)
    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks in the markdown viewer by opening them in the system browser."""
        import webbrowser

        try:
            webbrowser.open(event.href)
        except Exception as e:
            logger.warning(f"Failed to open link {event.href}: {e}")

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

    def action_show_info(self) -> None:
        """Show the info modal with app details and sponsor information."""
        self.push_screen(InfoModal())

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
        if "󰔛" in str(label.renderable):
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
