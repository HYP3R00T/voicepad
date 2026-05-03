"""VoicePad Textual TUI application."""

from __future__ import annotations

import logging
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Checkbox,
    Markdown,
    OptionList,
    Select,
    TabbedContent,
)
from voicepad_core import (
    ChunkResult,
    StreamingTranscriber,
    get_config,
)
from voicepad_core.config import Config

from voicepad.tui.managers import (
    LayoutBuilder,
    LifecycleManager,
    ModelManager,
    TabManager,
    TimerManager,
)
from voicepad.tui.modals import InfoModal
from voicepad.tui.models import SessionEntry
from voicepad.tui.utils.markdown import format_markdown as _format_markdown  # noqa: F401 (re-exported for tests)
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
        self._timer_thread: Any = None
        self._warm_result: ModelWarmResult | None = None
        self._current_text: str = ""
        self._selected_entry_idx: int | None = None
        self._hotkey_listener: Any = None  # GlobalHotkeyListener
        self._hotkey_pending_copy: bool = False
        self._overlay: Any = None  # StatusOverlay

        # Initialize managers
        self._layout_builder = LayoutBuilder(self, _APP_VERSION)
        self._lifecycle_manager = LifecycleManager(self)
        self._model_manager = ModelManager(self)
        self._timer_manager = TimerManager(self)
        self._tab_manager = TabManager(self)

        # Initialize handlers
        from voicepad.tui.handlers.history_handler import HistoryHandler
        from voicepad.tui.handlers.hotkey_handler import HotkeyHandler
        from voicepad.tui.handlers.recording_handler import RecordingHandler
        from voicepad.tui.handlers.settings_handler import SettingsHandler

        self._settings_handler = SettingsHandler(self)
        self._recording_handler = RecordingHandler(self)
        self._history_handler = HistoryHandler(self)
        self._hotkey_handler = HotkeyHandler(self)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield from self._layout_builder.compose()

    def on_mount(self) -> None:
        """Initialize the app after mounting."""
        self._layout_builder.mount_widgets()
        self._lifecycle_manager.on_mount()

    def on_unmount(self) -> None:
        """Clean up resources when the app exits."""
        self._lifecycle_manager.on_unmount()

    # ------------------------------------------------------------------
    # Global hotkey listener
    # ------------------------------------------------------------------

    def _start_hotkey_listener(self) -> None:
        """Start the system-wide hotkey listener and status overlay."""
        self._hotkey_handler.start_hotkey_listener()

    def _hotkey_on_start(self) -> None:
        """Called from the hotkey thread when the hotkey is pressed to start."""
        self._hotkey_handler.hotkey_on_start()

    def _hotkey_on_stop(self) -> None:
        """Called from the hotkey thread when the hotkey is pressed to stop."""
        self._hotkey_handler.hotkey_on_stop()

    def _hotkey_start_recording(self) -> None:
        """Start recording triggered by global hotkey (runs on main thread)."""
        self._hotkey_handler.hotkey_start_recording()

    def _hotkey_stop_recording(self) -> None:
        """Stop recording triggered by global hotkey (runs on main thread)."""
        self._hotkey_handler.hotkey_stop_recording()

    def _overlay_set(self, state: str) -> None:
        """Update the floating overlay state if it exists."""
        self._hotkey_handler.overlay_set(state)

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------

    def _populate_settings(self) -> None:
        """Build the settings form — only user-facing fields shown."""
        self._settings_handler.populate_settings()

    def _get_hotkey_from_picker(self) -> str:
        """Read modifier checkboxes + key dropdown and return pynput hotkey string."""
        return self._settings_handler.get_hotkey_from_picker()

    def _refresh_settings_values(self) -> None:
        """Update settings form widget values to match the current config in-place."""
        self._settings_handler.refresh_settings_values()

    def _refresh_config_path_label(self) -> None:
        """Update the settings config path label to reflect current file state."""
        self._settings_handler.refresh_config_path_label()

    @on(Checkbox.Changed, ".hotkey-checkbox")
    def on_hotkey_checkbox_changed(self) -> None:
        self._settings_handler.on_hotkey_checkbox_changed()

    @on(Select.Changed, "#hotkey-key-select")
    def on_hotkey_key_changed(self) -> None:
        self._settings_handler.on_hotkey_key_changed()

    @on(Button.Pressed, "#settings-save-btn")
    def on_settings_save(self) -> None:
        """Read user-facing inputs, merge with existing config, write to voicepad.yaml."""
        self._settings_handler.on_settings_save()

    # ------------------------------------------------------------------
    # Model warm-up
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=True, name="model-warm")
    def _warm_model_worker_impl(self) -> None:
        """Warm up the transcription model in a background thread (implementation)."""
        from voicepad.tui.workers import warm_model

        result = warm_model(self.config)
        self.call_from_thread(self._model_manager.on_model_ready, result)

    def _warm_model_worker(self) -> None:
        """Warm up the transcription model in a background thread."""
        self._warm_model_worker_impl()

    def _set_status(self, state: str, message: str) -> None:
        """Update the status label with icon and message."""
        self._model_manager.set_status(state, message)

    # ------------------------------------------------------------------
    # Tab-aware binding gating
    # ------------------------------------------------------------------

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Refresh footer bindings whenever the active tab changes."""
        self._tab_manager.on_tab_activated(event)

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Show/enable bindings only for the relevant tab and context."""
        return self._tab_manager.check_action(action, parameters)

    # ------------------------------------------------------------------
    # Record / stop
    # ------------------------------------------------------------------

    def action_toggle_recording(self) -> None:
        self._recording_handler.action_toggle_recording()

    def _start_recording(self) -> None:
        self._recording_handler.start_recording()

    def _stop_recording(self) -> None:
        self._recording_handler.stop_recording()

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    @work(thread=True, name="finalize")
    def _finalize_worker(self, audio: np.ndarray) -> None:
        """Stop the streamer (transcribes tail) then save the full recording."""
        self._recording_handler.finalize_worker(audio)

    def _on_stream_chunk(self, chunk: ChunkResult) -> None:
        """Called from the streaming thread for each transcribed chunk."""
        self._recording_handler.on_stream_chunk(chunk)

    def _save_recording(self, audio: np.ndarray) -> None:
        """Save WAV + markdown and add history entry after streaming completes."""
        self._recording_handler.save_recording(audio)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _load_history_from_disk(self) -> None:
        self._history_handler.load_history_from_disk()

    def _add_history_entry(self, entry: SessionEntry) -> None:
        self._history_handler.add_history_entry(entry)

    @on(OptionList.OptionSelected, "#history-options")
    def on_history_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._history_handler.on_history_option_selected(event)

    def _load_history_viewer(self, md_path: Path) -> None:
        self._history_handler.load_history_viewer(md_path)

    @on(Markdown.LinkClicked)
    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks in the markdown viewer by opening them in the system browser."""
        self._history_handler.on_markdown_link_clicked(event)

    # ------------------------------------------------------------------
    # Retranscribe (history tab)
    # ------------------------------------------------------------------

    @work(thread=True, name="retranscribe")
    def _retranscribe_file(self, wav_path: Path, md_path: Path | None) -> None:
        self._history_handler.retranscribe_file(wav_path, md_path)

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        self._timer_manager.start_timer()

    def _stop_timer(self) -> None:
        self._timer_manager.stop_timer()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_show_info(self) -> None:
        """Show the info modal with app details and sponsor information."""
        self.push_screen(InfoModal())

    def action_reload_model(self) -> None:
        """Re-download (if needed) and reload the current model."""
        self._model_manager.reload_model()

    def action_delete_entry(self) -> None:
        """Show delete confirmation for the selected history entry."""
        self._history_handler.action_delete_entry()

    def action_retranscribe_entry(self) -> None:
        """Retranscribe the selected history entry via keyboard shortcut."""
        self._history_handler.action_retranscribe_entry()

    def action_save_settings(self) -> None:
        """Save settings via keyboard shortcut."""
        self.query_one("#settings-save-btn", Button).press()

    def action_copy_transcription(self) -> None:
        self._history_handler.action_copy_transcription()

    @on(Button.Pressed, "#tx-copy-btn")
    def on_copy_btn_pressed(self) -> None:
        self._history_handler.on_copy_btn_pressed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the VoicePad TUI."""
    config = get_config()
    app = VoicePadApp(config)
    app.run()
