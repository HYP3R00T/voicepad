"""VoicePad Textual TUI application."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Input,
    Label,
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

from voicepad.tui.modals import DeleteConfirmModal, InfoModal, SetupModal
from voicepad.tui.models import SessionEntry
from voicepad.tui.theme import CATPPUCCIN_MOCHA_BLUE as _CATPPUCCIN_MOCHA_BLUE
from voicepad.tui.theme import MD_PLACEHOLDER as _MD_PLACEHOLDER
from voicepad.tui.theme import THEME_NAME as _THEME_NAME
from voicepad.tui.utils.clipboard import copy_to_clipboard as _copy_to_clipboard
from voicepad.tui.utils.hotkey_utils import HOTKEY_KEYS as _HOTKEY_KEYS
from voicepad.tui.utils.hotkey_utils import build_hotkey_str as _build_hotkey_str
from voicepad.tui.utils.hotkey_utils import parse_hotkey_str as _parse_hotkey_str
from voicepad.tui.utils.markdown import format_markdown as _format_markdown  # noqa: F401 (re-exported for tests)
from voicepad.tui.utils.markdown import format_markdown_streaming as _format_markdown_streaming
from voicepad.tui.utils.markdown import parse_markdown_entry as _parse_markdown_entry
from voicepad.tui.utils.markdown import prepend_retranscription as _prepend_retranscription
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
        self._hotkey_listener: Any = None  # GlobalHotkeyListener
        self._hotkey_pending_copy: bool = False
        self._overlay: Any = None  # StatusOverlay

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
        tx.mount(Button("\U000f0191  copy", id="tx-copy-btn", disabled=True))

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
                self._overlay.set_state(state)  # type: ignore[union-attr]

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

            def _setup_callback(result: tuple[str, int | None] | None) -> None:
                if result is not None:
                    self._on_setup_done(result)

            self.push_screen(SetupModal(self.config), callback=_setup_callback)
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
            # Sync hotkey picker from config
            mods, key = _parse_hotkey_str(self.config.global_hotkey)
            for mod_id in ("ctrl", "alt", "shift", "cmd"):
                self.query_one(f"#hotkey-mod-{mod_id}", Checkbox).value = mod_id in mods
            sel = self.query_one("#hotkey-key-select", Select)
            if key in _HOTKEY_KEYS:
                sel.value = key
            self._update_hotkey_preview()

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

        # ── Hotkey picker ──────────────────────────────────────────────
        hotkey_label = Label(
            "[bold]global_hotkey[/]  [dim]—  System-wide record/stop shortcut[/]",
            classes="settings-key",
        )
        hotkey_row = Static(classes="settings-row", id="hotkey-row")
        container.mount(hotkey_row)
        hotkey_row.mount(hotkey_label)

        # Parse current hotkey into modifiers + key
        mods, key_char = _parse_hotkey_str(self.config.global_hotkey)

        mod_row = Static(classes="hotkey-mod-row", id="hotkey-mod-row")
        hotkey_row.mount(mod_row)

        _modifiers = [("Ctrl", "ctrl"), ("Alt", "alt"), ("Shift", "shift"), ("Win", "cmd")]
        for label_text, mod_id in _modifiers:
            cb = Checkbox(
                label_text,
                value=(mod_id in mods),
                id=f"hotkey-mod-{mod_id}",
                classes="hotkey-checkbox",
            )
            mod_row.mount(cb)

        key_options = [(k, k) for k in _HOTKEY_KEYS]
        current_key = key_char if key_char in _HOTKEY_KEYS else "v"
        key_select = Select(
            options=key_options,
            value=current_key,
            id="hotkey-key-select",
            classes="hotkey-key-select",
            allow_blank=False,
        )
        hotkey_row.mount(key_select)

        preview = _build_hotkey_str(mods, current_key)
        hotkey_row.mount(
            Label(
                f"[dim]{preview or 'disabled'}[/]",
                id="hotkey-preview",
                classes="hotkey-preview",
            )
        )

    def _get_hotkey_from_picker(self) -> str:
        """Read modifier checkboxes + key dropdown and return pynput hotkey string."""
        mods: list[str] = []
        for mod_id in ("ctrl", "alt", "shift", "cmd"):
            with contextlib.suppress(Exception):
                if self.query_one(f"#hotkey-mod-{mod_id}", Checkbox).value:
                    mods.append(mod_id)
        key = "v"
        with contextlib.suppress(Exception):
            sel = self.query_one("#hotkey-key-select", Select)
            if sel.value is not Select.BLANK:
                key = str(sel.value)
        return _build_hotkey_str(mods, key)

    def _update_hotkey_preview(self) -> None:
        """Refresh the preview label from current picker state."""
        with contextlib.suppress(Exception):
            preview = self._get_hotkey_from_picker()
            self.query_one("#hotkey-preview", Label).update(f"[dim]{preview or 'disabled'}[/]")

    @on(Checkbox.Changed, ".hotkey-checkbox")
    def on_hotkey_checkbox_changed(self) -> None:
        self._update_hotkey_preview()

    @on(Select.Changed, "#hotkey-key-select")
    def on_hotkey_key_changed(self) -> None:
        self._update_hotkey_preview()

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

        # Read global hotkey from the picker widgets
        raw["global_hotkey"] = self._get_hotkey_from_picker()

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
                        v = int(str(sel.value))
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
            status.update(f"[red]\U000f0156  {'; '.join(errors)}[/]")
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
                        self._hotkey_listener.stop()  # type: ignore[union-attr]
                if self._overlay is not None:
                    with contextlib.suppress(Exception):
                        self._overlay.stop()  # type: ignore[union-attr]
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
                status.update("[green]\U000f012c  saved — reloading model[/]")
            else:
                status.update("[green]\U000f012c  saved[/]")

            self.set_timer(3.0, lambda: status.update(""))
        except Exception as e:
            status.update(f"[red]\U000f0156  {e}[/]")

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
        recorder = self._session._recorder
        if recorder is None:
            return
        self._streamer = StreamingTranscriber(
            recorder=recorder,
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
            idx = int(event.option.id or "-1")
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

        def _delete_callback(result: bool | None) -> None:
            if result is not None:
                self._on_delete_confirmed(result)

        self.push_screen(DeleteConfirmModal(name), callback=_delete_callback)

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
            btn.label = "\U000f012c  copied"
            self.set_timer(1.5, lambda: setattr(btn, "label", "\U000f0191  copy"))

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
        if "󰔛" in str(label.render()):
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
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the VoicePad TUI."""
    config = get_config()
    app = VoicePadApp(config)
    app.run()
