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
    DataTable,
    Footer,
    Input,
    Label,
    Link,
    Markdown,
    MarkdownViewer,
    OptionList,
    ProgressBar,
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
            yield Static(
                f"[dim]{self._entry_name}[/]",
                id="delete-name",
            )
            yield Static(
                "This will permanently delete the WAV file\nand its transcription markdown.",
                id="delete-body",
            )
            with Static(id="delete-nav"):
                yield Button("Cancel", id="delete-cancel")
                yield Static("", id="delete-spacer")
                yield Button("Delete", id="delete-confirm")

    def action_dismiss_false(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#delete-cancel")
    def on_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#delete-confirm")
    def on_confirm(self) -> None:
        self.dismiss(True)


class SetupModal(ModalScreen[tuple[str, int | None]]):
    """4-step first-run wizard.

    Step 0 — Welcome
    Step 1 — Model selection + download
    Step 2 — Microphone selection
    Step 3 — Finish / config info
    """

    BINDINGS: list[Binding] = []  # no escape — must complete setup

    _HINTS: dict[str, str] = {
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

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._step: int = 0
        self._chosen_model: str = config.transcription_model
        self._chosen_device_index: int | None = config.input_device_index
        self._downloading = False

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Static(id="wizard-dialog"):
            yield Static("", id="wizard-step-indicator")
            yield Static(id="wizard-body")
            with Static(id="wizard-nav"):
                yield Button("← Back", id="wizard-back", variant="default")
                yield Static("", id="wizard-spacer")
                yield Button("Next →", id="wizard-next", variant="primary")

    def on_mount(self) -> None:
        self._render_step()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#wizard-next")
    def on_next(self) -> None:
        if self._step == 1 and not self._downloading:
            # Must download before advancing from step 1
            self._start_download()
            return
        if self._step < 3:
            self._step += 1
            self._render_step()
        else:
            # Finish
            self.app.call_after_refresh(self._do_dismiss)

    @on(Button.Pressed, "#wizard-back")
    def on_back(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._render_step()

    def _do_dismiss(self) -> None:
        self.dismiss((self._chosen_model, self._chosen_device_index))

    # ------------------------------------------------------------------
    # Step renderer
    # ------------------------------------------------------------------

    def _render_step(self) -> None:
        self.query_one("#wizard-step-indicator", Static).update(f"[dim]Step {self._step + 1} of 4[/]")
        body = self.query_one("#wizard-body", Static)
        # Remove all children and remount for the current step
        for child in list(body.children):
            child.remove()

        back_btn = self.query_one("#wizard-back", Button)
        next_btn = self.query_one("#wizard-next", Button)
        back_btn.display = self._step > 0
        next_btn.label = "Finish" if self._step == 3 else ("Download & Continue →" if self._step == 1 else "Next →")
        next_btn.disabled = False

        if self._step == 0:
            self._mount_step_welcome(body)
        elif self._step == 1:
            self._mount_step_model(body)
        elif self._step == 2:
            self._mount_step_microphone(body)
        elif self._step == 3:
            self._mount_step_finish(body)

        # Always focus the primary action button — never leave focus on Back
        self.set_timer(0.05, lambda: next_btn.focus())

    # ------------------------------------------------------------------
    # Step 0 — Welcome
    # ------------------------------------------------------------------

    def _mount_step_welcome(self, body: Static) -> None:
        body.mount(Static(f"󰍬  VoicePad  {_APP_VERSION}", classes="wizard-title"))
        body.mount(
            Static(
                "Your private, local-first dictation studio.",
                classes="wizard-text",
            )
        )
        body.mount(
            Static(
                "󰒍  100% local  ·  󰌨  GPU-accelerated  ·  󰍹  no cloud",
                classes="wizard-features",
            )
        )
        body.mount(
            Static(
                "[dim]by Rajesh Das (HYP3R00T)  ·  MIT License[/]",
                classes="wizard-credit",
            )
        )

    # ------------------------------------------------------------------
    # Step 1 — Model selection + download
    # ------------------------------------------------------------------

    def _mount_step_model(self, body: Static) -> None:
        body.mount(Static("Choose a Whisper Model", classes="wizard-title"))
        body.mount(
            Static(
                "[dim]turbo[/] is recommended for most NVIDIA GPU users.",
                classes="wizard-text",
            )
        )
        body.mount(
            Select(
                options=[(m, m) for m in VALID_TRANSCRIPTION_MODELS],
                value=self._chosen_model,
                id="wizard-model-select",
                allow_blank=False,
            )
        )
        body.mount(Static("", id="wizard-model-hint", classes="wizard-hint"))
        body.mount(Static("", id="wizard-download-status", classes="wizard-status"))
        bar = ProgressBar(id="wizard-progress", show_eta=False, show_percentage=True)
        bar.display = False
        body.mount(bar)

    @on(Select.Changed, "#wizard-model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._chosen_model = str(event.value)
            with contextlib.suppress(Exception):
                self.query_one("#wizard-model-hint", Static).update(self._HINTS.get(self._chosen_model, ""))

    def _start_download(self) -> None:
        if self._downloading:
            return
        self._downloading = True
        self.query_one("#wizard-next", Button).disabled = True
        self._download_model_worker(self._chosen_model)

    @work(thread=True, name="setup-download")
    def _download_model_worker(self, model: str) -> None:
        from voicepad_core import ensure_model_downloaded, model_downloaded
        from voicepad_core.transcription import TranscriptionError

        self.app.call_from_thread(self._set_download_status, f"Checking '{model}'…")

        if model_downloaded(model, self._config):
            self.app.call_from_thread(self._on_download_done, model, None)
            return

        self.app.call_from_thread(self._set_download_status, f"Downloading '{model}'…")
        self.app.call_from_thread(self._show_progress_bar)

        _last_pct = [-1]  # mutable container for closure

        def _on_progress(downloaded: int, total: int) -> None:
            if total > 0:
                pct = min(100, int(downloaded * 100 / total))
                if pct == _last_pct[0]:
                    return  # skip — no change, don't flood the event loop
                _last_pct[0] = pct
            self.app.call_from_thread(self._update_progress, downloaded, total)

        try:
            ensure_model_downloaded(model, self._config, on_progress=_on_progress)
            self.app.call_from_thread(self._on_download_done, model, None)
        except TranscriptionError as e:
            self.app.call_from_thread(self._on_download_done, model, str(e))

    def _show_progress_bar(self) -> None:
        with contextlib.suppress(Exception):
            bar = self.query_one("#wizard-progress", ProgressBar)
            bar.display = True
            bar.update(total=100, progress=0)
        self._last_pct: int = -1

    def _update_progress(self, downloaded: int, total: int) -> None:
        with contextlib.suppress(Exception):
            mb_done = downloaded / 1_048_576
            if total > 0:
                pct = min(100, int(downloaded * 100 / total))
                # Only update UI when percentage actually changes
                if pct == getattr(self, "_last_pct", -1):
                    return
                self._last_pct = pct
                mb_total = total / 1_048_576
                self.query_one("#wizard-progress", ProgressBar).update(total=100, progress=pct)
                self._set_download_status(f"Downloading… {mb_done:.0f} / {mb_total:.0f} MB")
            else:
                self._set_download_status(f"Downloading… {mb_done:.1f} MB")

    def _set_download_status(self, msg: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#wizard-download-status", Static).update(msg)

    def _on_download_done(self, model: str, error: str | None) -> None:
        self._downloading = False
        with contextlib.suppress(Exception):
            self.query_one("#wizard-progress", ProgressBar).display = False
        if error:
            self._set_download_status(f"[red]✕  Download failed: {error}[/]")
            self.query_one("#wizard-next", Button).disabled = False
        else:
            self._set_download_status(f"[green]✓  '{model}' ready[/]")
            self._step += 1
            self.set_timer(0.6, self._render_step)

    # ------------------------------------------------------------------
    # Step 2 — Microphone
    # ------------------------------------------------------------------

    def _mount_step_microphone(self, body: Static) -> None:
        from voicepad.cli.config import _get_input_devices

        body.mount(Static("Select Your Microphone", classes="wizard-title"))
        body.mount(
            Static(
                "[dim]System default[/] works for most setups.",
                classes="wizard-text",
            )
        )

        devices = _get_input_devices()
        device_options: list[tuple[str, int]] = [("System default", -1)]
        device_options += [(f"[{d.index}]  {d.name}", d.index) for d in devices]

        current = self._chosen_device_index if self._chosen_device_index is not None else -1
        valid = {v for _, v in device_options}

        body.mount(
            Select(
                options=device_options,
                value=current if current in valid else -1,
                id="wizard-device-select",
                allow_blank=False,
            )
        )
        body.mount(
            Static(
                f"[dim]{len(devices)} input device(s) found[/]",
                classes="wizard-hint",
            )
        )

    @on(Select.Changed, "#wizard-device-select")
    def on_device_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            v = int(event.value)
            self._chosen_device_index = None if v == -1 else v

    # ------------------------------------------------------------------
    # Step 3 — Finish
    # ------------------------------------------------------------------

    def _mount_step_finish(self, body: Static) -> None:
        from utilityhub_config import get_config_path

        config_path = get_config_path("voicepad", format="yaml")
        body.mount(Static("󰄬  You're all set!", classes="wizard-title"))
        body.mount(
            Static(
                "VoicePad is ready. Here are the essentials:",
                classes="wizard-text",
            )
        )

        table = DataTable(
            id="wizard-keybindings",
            show_header=False,
            show_cursor=False,
            zebra_stripes=False,
        )
        body.mount(table)
        table.add_columns("key", "action")
        table.add_rows([
            ("Space", "start / stop recording"),
            ("c", "copy transcription to clipboard"),
            ("i", "info & links"),
            ("q", "quit"),
        ])

        body.mount(
            Static(
                f"Tweak anything in the [bold]Settings[/] tab.\n[dim]{config_path}[/]",
                classes="wizard-hint",
            )
        )


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
        # Global
        Binding("q", "quit", "Quit", show=True),
        Binding("i", "show_info", "Info", show=True, key_display="i"),
        # Record tab
        Binding("space", "toggle_recording", "Record / Stop", show=True),
        Binding("c", "copy_transcription", "Copy", show=True),
        # History tab
        Binding("t", "retranscribe_entry", "Retranscribe", show=True),
        Binding("d", "delete_entry", "Delete", show=True),
        # Settings tab
        Binding("s", "save_settings", "Save", show=True),
        # Hidden utility
        Binding("r", "reload_model", "Reload model", show=False),
    ]

    _model_ready: reactive[bool] = reactive(False)
    _recording: reactive[bool] = reactive(False)
    _transcribing: reactive[bool] = reactive(False)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._session: RecordingSession | None = None
        self._streamer: StreamingTranscriber | None = None
        self._stream_chunks: list[ChunkResult] = []
        self._entries: list[SessionEntry] = []
        self._record_start: float = 0.0
        self._timer_thread: threading.Thread | None = None
        self._warm_result: ModelWarmResult | None = None
        self._current_text: str = ""
        self._selected_entry_idx: int | None = None
        self._hotkey_listener: object | None = None  # GlobalHotkeyListener
        self._hotkey_pending_copy: bool = False
        self._overlay: object | None = None  # StatusOverlay

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

                with Static(id="history-view-pane") as view_pane:
                    view_pane.border_title = "transcription"
                    yield MarkdownViewer(
                        _MD_PLACEHOLDER,
                        id="history-viewer",
                        show_table_of_contents=False,
                        open_links=False,
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

    def on_unmount(self) -> None:
        """Stop the global hotkey listener when the app exits."""
        if self._hotkey_listener is not None:
            with contextlib.suppress(Exception):
                self._hotkey_listener.stop()
        if self._overlay is not None:
            with contextlib.suppress(Exception):
                self._overlay.stop()

    # ------------------------------------------------------------------
    # Global hotkey listener
    # ------------------------------------------------------------------

    def _start_hotkey_listener(self) -> None:
        """Start the system-wide hotkey listener and status overlay."""
        hotkey = getattr(self.config, "global_hotkey", "")
        if not hotkey:
            return
        try:
            from voicepad.tui.hotkey import GlobalHotkeyListener
            from voicepad.tui.overlay import StatusOverlay

            self._overlay = StatusOverlay()
            self._overlay.start()

            self._hotkey_listener = GlobalHotkeyListener(
                hotkey=hotkey,
                on_start=self._hotkey_on_start,
                on_stop=self._hotkey_on_stop,
            )
            self._hotkey_listener.start()
            logger.info(f"Global hotkey active: {hotkey}")
        except Exception as e:
            logger.warning(f"Could not start global hotkey listener: {e}")

    def _hotkey_on_start(self) -> None:
        """Called from the hotkey thread when the hotkey is pressed to start."""
        self.call_from_thread(self._hotkey_start_recording)

    def _hotkey_on_stop(self) -> None:
        """Called from the hotkey thread when the hotkey is pressed to stop."""
        self.call_from_thread(self._hotkey_stop_recording)

    def _hotkey_start_recording(self) -> None:
        """Start recording triggered by global hotkey (runs on main thread)."""
        if self._recording or self._transcribing:
            return
        if not self._model_ready:
            self._overlay_set("error")
            return
        # Switch to record tab so the user can see what's happening
        with contextlib.suppress(Exception):
            self.query_one("#tabs", TabbedContent).active = "tab-record"
        self._overlay_set("recording")
        self._start_recording()

    def _hotkey_stop_recording(self) -> None:
        """Stop recording triggered by global hotkey (runs on main thread)."""
        if not self._recording:
            return
        self._hotkey_pending_copy = True  # flag to auto-copy after transcription
        self._overlay_set("transcribing")
        self._stop_recording()

    def _overlay_set(self, state: str) -> None:
        """Update the floating overlay state if it exists."""
        if self._overlay is not None:
            with contextlib.suppress(Exception):
                self._overlay.set_state(state)

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
                SetupModal(self.config),
                callback=self._on_setup_done,
            )
        else:
            self._warm_model_worker()

    def _on_setup_done(self, result: tuple[str, int | None]) -> None:
        """Called when the setup wizard finishes. Writes config with chosen model + device."""
        from utilityhub_config import get_config_path, write_config
        from voicepad_core.config import Config as _Config

        chosen_model, chosen_device = result
        raw = self.config.model_dump(mode="json")
        raw["transcription_model"] = chosen_model
        raw["input_device_index"] = chosen_device

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
        self._refresh_settings_values()
        self._warm_model_worker()

    def _refresh_settings_values(self) -> None:
        """Update settings form widget values to match the current config in-place."""
        with contextlib.suppress(Exception):
            from voicepad.cli.config import _get_input_devices

            # Update device dropdown
            devices = _get_input_devices()
            device_options: list[tuple[str, int]] = [("system default", -1)]
            device_options += [(f"[{d.index}]  {d.name}", d.index) for d in devices]
            valid = {v for _, v in device_options}
            current_idx = self.config.input_device_index if self.config.input_device_index is not None else -1
            sel_device = self.query_one("#setting-input_device_index", Select)
            sel_device.set_options(device_options)
            sel_device.value = current_idx if current_idx in valid else -1

        with contextlib.suppress(Exception):
            # Update model dropdown
            sel_model = self.query_one("#setting-transcription_model", Select)
            sel_model.value = self.config.transcription_model

        with contextlib.suppress(Exception):
            # Update path inputs
            from textual.widgets import Input as _Input

            self.query_one("#setting-recordings_path", _Input).value = str(self.config.recordings_path)
            self.query_one("#setting-markdown_path", _Input).value = str(self.config.markdown_path)

        with contextlib.suppress(Exception):
            from textual.widgets import Input as _Input2

            self.query_one("#setting-global_hotkey", _Input2).value = self.config.global_hotkey

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
            "global_hotkey": "System-wide hotkey (e.g. <ctrl>+<alt>+v, empty to disable)",
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

            hotkey_changed = new_config.global_hotkey != self.config.global_hotkey
            model_changed = (
                new_config.transcription_model != self.config.transcription_model
                or new_config.transcription_device != self.config.transcription_device
                or new_config.transcription_compute_type != self.config.transcription_compute_type
            )
            object.__setattr__(self, "config", new_config)

            if hotkey_changed:
                if self._hotkey_listener is not None:
                    with contextlib.suppress(Exception):
                        self._hotkey_listener.stop()
                if self._overlay is not None:
                    with contextlib.suppress(Exception):
                        self._overlay.stop()
                self._hotkey_listener = None
                self._overlay = None
                self._start_hotkey_listener()

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

        # Start the global hotkey listener now that the model is ready
        # and the Textual event loop is fully running.
        if self._hotkey_listener is None:
            self._start_hotkey_listener()

    # ------------------------------------------------------------------
    # Tab-aware binding gating
    # ------------------------------------------------------------------

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Refresh footer bindings whenever the active tab changes."""
        # Auto-select the latest history entry when switching to history tab
        if str(event.tab.id) == "tab-history" and self._entries and self._selected_entry_idx is None:
            ol = self.query_one("#history-options", OptionList)
            if ol.option_count > 0:
                ol.highlighted = ol.option_count - 1
                last_entry = self._entries[-1]
                self._selected_entry_idx = last_entry.index
                if last_entry.md_path and last_entry.md_path.exists():
                    self._load_history_viewer(last_entry.md_path)
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Show/enable bindings only for the relevant tab and context."""
        active = self.query_one("#tabs", TabbedContent).active if self.is_mounted else "tab-record"
        tab_specific: dict[str, str] = {
            "toggle_recording": "tab-record",
            "copy_transcription": "tab-record",
            "retranscribe_entry": "tab-history",
            "delete_entry": "tab-history",
            "save_settings": "tab-settings",
        }
        if action in tab_specific:
            if active != tab_specific[action]:
                return False
            # t and d also require an entry to be selected
            if action in ("retranscribe_entry", "delete_entry"):
                return self._selected_entry_idx is not None
        return True

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
            # Auto-copy if triggered by global hotkey
            if self._hotkey_pending_copy:
                self._hotkey_pending_copy = False
                full_text = " ".join(c.text for c in self._stream_chunks).strip()
                if full_text:
                    _copy_to_clipboard(full_text)
                    self._set_status("ready", "ready — copied to clipboard")
                    self._overlay_set("copied")
                else:
                    self._overlay_set("hidden")

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
                    _format_markdown_streaming(
                        wav_path, full_text, duration_s, self._stream_chunks, self.config.transcription_model
                    ),
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
            idx = int(event.option.id)
            if 0 <= idx < len(self._entries):
                self._selected_entry_idx = idx
                entry = self._entries[idx]
                self.refresh_bindings()
                if entry.md_path and entry.md_path.exists():
                    self._load_history_viewer(entry.md_path)

    @work(name="md-view")
    async def _load_history_viewer(self, md_path: Path) -> None:
        viewer = self.query_one("#history-viewer", MarkdownViewer)
        try:
            raw = md_path.read_text(encoding="utf-8")
            lines = raw.splitlines()

            # Parse YAML front matter into per-transcription metadata
            fm_meta: dict[int, dict] = {}
            wav_name = ""
            body_lines: list[str] = lines

            if lines and lines[0].strip() == "---":
                fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
                if fm_end is not None:
                    current: dict | None = None
                    for fl in lines[1:fm_end]:
                        s = fl.strip()
                        if s.startswith("file:"):
                            wav_name = s.split(":", 1)[-1].strip()
                        elif s.startswith("- n:"):
                            with contextlib.suppress(Exception):
                                current = {"n": int(s.split(":")[-1].strip())}
                        elif current is not None and ":" in s:
                            k, _, v = s.partition(":")
                            current[k.strip()] = v.strip()
                            fm_meta[current["n"]] = current
                    body_lines = lines[fm_end + 1 :]

            # Rebuild display content: inject metadata after each ## Transcription N heading
            out: list[str] = []
            if wav_name:
                out += [f"**File:** `{wav_name}`", ""]

            for line in body_lines:
                stripped = line.strip()
                out.append(line)
                if stripped.startswith("## Transcription "):
                    with contextlib.suppress(Exception):
                        n = int(stripped.split()[-1])
                        meta = fm_meta.get(n, {})
                        if meta:
                            parts = []
                            if "model" in meta:
                                parts.append(f"model: {meta['model']}")
                            if "language" in meta:
                                parts.append(f"language: {meta['language']}")
                            if "duration" in meta:
                                parts.append(f"duration: {meta['duration']}")
                            if "latency" in meta:
                                parts.append(f"latency: {meta['latency']}")
                            if "timestamp" in meta:
                                parts.append(f"_{meta['timestamp']}_")
                            if parts:
                                out.append("")
                                out.append("*" + " · ".join(parts) + "*")

            await viewer.document.update("\n".join(out))
        except Exception:
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
        if error:
            self._set_status("error", error)
            return

        if result:
            # Prepend new transcription — never overwrite the existing ones
            out_md = md_path or (self.config.markdown_path / f"{wav_path.stem}.md")
            self.config.markdown_path.mkdir(parents=True, exist_ok=True)
            new_content = _prepend_retranscription(out_md, result, self.config.transcription_model)
            out_md.write_text(new_content, encoding="utf-8")
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

    def action_reload_model(self) -> None:
        """Re-download (if needed) and reload the current model."""
        if self._recording or self._transcribing:
            return
        from voicepad_core.transcription import _model_cache

        _model_cache.clear()
        self._model_ready = False
        self._set_status("transcribing", "reloading model…")
        self.query_one("#header-model", Label).update("[dim]model:[/] loading…")
        self._warm_model_worker()

    def action_delete_entry(self) -> None:
        """Show delete confirmation for the selected history entry."""
        active = self.query_one("#tabs", TabbedContent).active
        if active != "tab-history" or self._selected_entry_idx is None:
            return
        self._show_delete_confirm()

    def action_retranscribe_entry(self) -> None:
        """Retranscribe the selected history entry via keyboard shortcut."""
        if self._selected_entry_idx is None or not self._model_ready:
            return
        entry = self._entries[self._selected_entry_idx]
        if entry.wav_path and entry.wav_path.exists():
            self._retranscribe_file(entry.wav_path, entry.md_path)

    def action_save_settings(self) -> None:
        """Save settings via keyboard shortcut."""
        self.query_one("#settings-save-btn", Button).press()

    def _show_delete_confirm(self) -> None:
        if self._selected_entry_idx is None:
            return
        entry = self._entries[self._selected_entry_idx]
        name = entry.wav_path.stem if entry.wav_path else f"clip-{entry.index + 1}"
        self.push_screen(DeleteConfirmModal(name), callback=self._on_delete_confirmed)

    def _on_delete_confirmed(self, confirmed: bool) -> None:
        if not confirmed or self._selected_entry_idx is None:
            return
        entry = self._entries[self._selected_entry_idx]

        # Delete files from disk
        with contextlib.suppress(Exception):
            if entry.wav_path and entry.wav_path.exists():
                entry.wav_path.unlink()
        with contextlib.suppress(Exception):
            if entry.md_path and entry.md_path.exists():
                entry.md_path.unlink()

        # Remove from in-memory list
        del self._entries[self._selected_entry_idx]
        self._selected_entry_idx = None

        # Rebuild the OptionList from scratch
        ol = self.query_one("#history-options", OptionList)
        ol.clear_options()
        for new_idx, e in enumerate(self._entries):
            e = SessionEntry(
                index=new_idx,
                wav_path=e.wav_path,
                md_path=e.md_path,
                duration_s=e.duration_s,
                text=e.text,
                latency_ms=e.latency_ms,
                device=e.device,
                timestamp=e.timestamp,
            )
            self._entries[new_idx] = e
            name = e.wav_path.stem if e.wav_path else f"clip-{new_idx + 1}"
            label = (
                f"[bold]{e.timestamp}[/]  [dim]{name}[/]\n"
                f"  [dim]{e.duration_s:.1f}s · {e.latency_ms:.0f}ms · {e.device}[/]"
            )
            ol.add_option(Option(label, id=str(new_idx)))

        # Clear the viewer and disable action buttons
        self.refresh_bindings()
        with contextlib.suppress(Exception):
            self.run_worker(
                self.query_one("#history-viewer", MarkdownViewer).document.update(_MD_PLACEHOLDER),
                name="md-clear",
            )

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


def _format_markdown(audio_path: Path, result, model_name: str = "") -> str:
    """Create a new markdown file with the first transcription."""
    ts = time.strftime("%Y-%m-%d %H:%M")
    model_str = (
        f"{model_name} · {result.device} / {result.compute_type}"
        if model_name
        else f"{result.device} / {result.compute_type}"
    )
    fm = [
        "---",
        f"file: {audio_path.name}",
        "transcriptions:",
        "  - n: 1",
        f"    model: {model_str}",
        f"    language: {result.language} ({result.language_probability * 100:.1f}%)",
        f"    duration: {result.duration_s:.1f}s",
        f"    latency: {result.latency_ms:.0f}ms",
        f"    timestamp: {ts}",
        "---",
        "",
        "## Transcription 1",
        "",
        result.text or "*(no speech detected)*",
        "",
    ]
    return "\n".join(fm)


def _format_markdown_streaming(
    wav_path: Path,
    text: str,
    duration_s: float,
    chunks: list[ChunkResult],
    model_name: str = "",
) -> str:
    """Create a new markdown file for a streaming transcription."""
    latest_chunk = next((chunk for chunk in reversed(chunks) if chunk.text), None)
    device = latest_chunk.device if latest_chunk else "unknown"
    language = latest_chunk.language if latest_chunk else "en"
    language_probability = latest_chunk.language_probability if latest_chunk else 0.0
    latency_ms = sum(chunk.latency_ms for chunk in chunks)
    ts = time.strftime("%Y-%m-%d %H:%M")
    model_str = f"{model_name} · {device} / live" if model_name else f"{device} / live"

    fm = [
        "---",
        f"file: {wav_path.name}",
        "transcriptions:",
        "  - n: 1",
        f"    model: {model_str}",
        f"    language: {language} ({language_probability * 100:.1f}%)",
        f"    duration: {duration_s:.1f}s",
        f"    latency: {latency_ms:.0f}ms",
        f"    timestamp: {ts}",
        "---",
        "",
        "## Transcription 1",
        "",
        text or "*(no speech detected)*",
        "",
    ]
    return "\n".join(fm)


def _prepend_retranscription(md_path: Path, result, model_name: str = "") -> str:
    """Prepend a new transcription to an existing markdown file.

    Reads the existing file, increments the transcription count,
    adds new metadata entry at the top of the array, and prepends
    the new text block before the existing ones.

    Returns the new file content.
    """
    ts = time.strftime("%Y-%m-%d %H:%M")

    try:
        existing = md_path.read_text(encoding="utf-8")
    except Exception:
        existing = ""

    # Parse existing front matter to find the current max n
    lines = existing.splitlines()
    max_n = 0
    if lines and lines[0].strip() == "---":
        fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
        if fm_end is not None:
            for fl in lines[1:fm_end]:
                if fl.strip().startswith("- n:"):
                    with contextlib.suppress(Exception):
                        max_n = max(max_n, int(fl.strip().split(":")[-1].strip()))

    new_n = max_n + 1

    # Build new front matter entry
    model_str = (
        f"{model_name} · {result.device} / {result.compute_type}"
        if model_name
        else f"{result.device} / {result.compute_type}"
    )
    new_fm_entry = [
        f"  - n: {new_n}",
        f"    model: {model_str}",
        f"    language: {result.language} ({result.language_probability * 100:.1f}%)",
        f"    duration: {result.duration_s:.1f}s",
        f"    latency: {result.latency_ms:.0f}ms",
        f"    timestamp: {ts}",
    ]

    # Inject new entry right after "transcriptions:" line in front matter
    new_lines: list[str] = []
    injected = False
    if lines and lines[0].strip() == "---":
        fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
        if fm_end is not None:
            for line in lines[: fm_end + 1]:
                new_lines.append(line)
                if not injected and line.strip() == "transcriptions:":
                    new_lines.extend(new_fm_entry)
                    injected = True
            body = lines[fm_end + 1 :]
        else:
            new_lines = lines
            body = []
    else:
        new_lines = ["---", f"file: {md_path.stem}.wav", "transcriptions:"] + new_fm_entry + ["---"]
        body = lines

    # Build new content: front matter + new transcription block + existing body
    new_block = [
        "",
        f"## Transcription {new_n}",
        "",
        result.text or "*(no speech detected)*",
        "",
    ]

    all_lines = new_lines + new_block + ([""] if body and body[0] != "" else []) + body
    return "\n".join(all_lines)


def _parse_markdown_entry(md_path: Path, index: int, recordings_path: Path | None = None) -> SessionEntry | None:
    """Parse a transcription markdown file (YAML front matter format) into a SessionEntry.

    Uses the latest transcription (highest n) for the preview text and metadata.
    """
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = content.splitlines()

    if not lines or lines[0].strip() != "---":
        return None

    fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
    if fm_end is None:
        return None

    wav_name: str | None = None
    # Parse transcriptions array — collect all entries, pick the one with highest n
    entries: list[dict] = []
    current: dict | None = None
    for fl in lines[1:fm_end]:
        stripped = fl.strip()
        if stripped.startswith("- n:"):
            if current is not None:
                entries.append(current)
            with contextlib.suppress(Exception):
                current = {"n": int(stripped.split(":")[-1].strip())}
        elif stripped.startswith("file:"):
            wav_name = stripped.split(":", 1)[-1].strip()
        elif current is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            current[key.strip()] = val.strip()
    if current is not None:
        entries.append(current)

    if not entries:
        return None

    latest = max(entries, key=lambda e: e.get("n", 0))
    duration_s = 0.0
    latency_ms = 0.0
    device = "unknown"
    with contextlib.suppress(Exception):
        duration_s = float(latest.get("duration", "0s").rstrip("s"))
    with contextlib.suppress(Exception):
        latency_ms = float(latest.get("latency", "0ms").rstrip("ms"))
    with contextlib.suppress(Exception):
        device = latest.get("model", "unknown").split("/")[0].strip()

    # Extract text for the latest transcription block
    latest_n = latest.get("n", 1)
    marker = f"## Transcription {latest_n}"
    body = lines[fm_end + 1 :]
    text_lines: list[str] = []
    in_block = False
    for line in body:
        if line.strip() == marker:
            in_block = True
            continue
        if in_block:
            # Stop at the next transcription marker
            if line.strip().startswith("## Transcription "):
                break
            text_lines.append(line)
    text = " ".join(ln for ln in text_lines if ln.strip() and ln.strip() != "*(no speech detected)*").strip()

    if not text:
        return None

    # Resolve WAV path
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

    timestamp = latest.get("timestamp", "")
    if not timestamp:
        parts = md_path.stem.split("_")
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
