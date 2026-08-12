from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Footer,
    Input,
    Label,
    Markdown,
    OptionList,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from voicepad_core.audio import MicrophoneStream
from voicepad_core.pipeline import (
    FileTranscriptionResult,
    GrowingTranscriptionJob,
    GrowingTranscriptionUpdate,
)

from voicepad.config import AppConfig, load_config, save_config
from voicepad.output import persist_markdown
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.components import VoiceButton
from voicepad.tui.control import ControlServer
from voicepad.tui.shortcut import desktop_shortcut_status, open_shortcut_settings
from voicepad.tui.status import DesktopStatus
from voicepad.tui.status import State as DesktopStatusState
from voicepad.tui.theme import THEMES
from voicepad.tui.utils.clipboard import copy_to_clipboard

try:
    APP_VERSION = f"v{version('voicepad')}"
except Exception:
    APP_VERSION = "dev"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    markdown: Path
    audio: Path | None
    duration_seconds: float
    complete: bool
    text: str


class VoicePadApp(App[None]):
    """Resident-model dictation UI with recording, history, and settings."""

    TITLE = "VoicePad"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("space", "toggle_recording", "Record / stop"),
        Binding("c", "copy", "Copy"),
        Binding("s", "save_settings", "Save settings"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, config: AppConfig | None = None, runtime: ApplicationRuntime | None = None) -> None:
        super().__init__(ansi_color=True)
        self.config = config or load_config()
        self.theme = self.config.theme
        self.runtime = runtime or ApplicationRuntime(self.config)
        self._state = "loading"
        self._microphone: MicrophoneStream | None = None
        self._job: GrowingTranscriptionJob | None = None
        self._last_result: FileTranscriptionResult | None = None
        self._record_started = 0.0
        self._finalization_lock = threading.Lock()
        self._history: list[HistoryEntry] = []
        self._desktop_status: DesktopStatus | None = None
        self._external_session = False
        self._shortcut = desktop_shortcut_status()
        self._control = ControlServer(lambda: self.call_from_thread(self._external_toggle))

    def compose(self) -> ComposeResult:
        with Static(id="header"):
            yield Label("VoicePad", id="header-title")
            yield Label(APP_VERSION, id="header-version")
            yield Label("󰔟  initialising", id="status")
            yield Label("loading…", id="header-model")
        yield Static(id="header-rule")
        with Static(id="body"), TabbedContent(id="tabs"):
            with TabPane("  record  ", id="tab-record"), Static(id="transcription") as transcript:
                transcript.border_title = "transcription"
                yield Label("speak and press space to begin…", id="tx-text", classes="placeholder")
                yield Label("", id="tx-meta")
                yield VoiceButton("󰆏  copy", id="tx-copy-btn", disabled=True)
            with TabPane("  history  ", id="tab-history"), Static(id="history-section"):
                with Static(id="history-list-pane") as history_list:
                    history_list.border_title = "recordings"
                    yield OptionList(id="history-options")
                with Static(id="history-view-pane") as history_view:
                    history_view.border_title = "transcript"
                    with Static(id="history-detail"):
                        yield Label("No recordings yet", id="history-detail-title")
                        yield Label("Your completed dictations will appear here.", id="history-detail-meta")
                    yield Markdown("_Record something to begin your local history._", id="history-viewer")
            with TabPane("  settings  ", id="tab-settings"):
                yield from self._compose_settings()
        yield Footer()

    def _compose_settings(self) -> ComposeResult:
        with VerticalScroll(id="settings-scroll"), Static(id="settings-fields"):
            with Static(classes="settings-group"):
                yield Label("Storage", classes="settings-section-title")
                yield Label(
                    "Choose where durable audio, results, and verified model files live.",
                    classes="settings-section-copy",
                )
                with Static(classes="settings-item"):
                    yield Label("Recordings", classes="settings-key")
                    yield Label("Canonical WAV audio", classes="settings-help")
                    yield Input(str(self.config.recordings_path), id="setting-recordings", classes="settings-input")
                with Static(classes="settings-item"):
                    yield Label("Transcripts", classes="settings-key")
                    yield Label("Schema-1 Markdown results", classes="settings-help")
                    yield Input(str(self.config.markdown_path), id="setting-markdown", classes="settings-input")
                with Static(classes="settings-item"):
                    yield Label("Model cache", classes="settings-key")
                    yield Label("Verified Parakeet and Silero artifacts", classes="settings-help")
                    yield Input(str(self.config.artifact_cache_path), id="setting-artifacts", classes="settings-input")

            with Static(classes="settings-group"):
                yield Label("Experience", classes="settings-section-title")
                yield Label(
                    "Tune naming, appearance, and what happens after a complete result.",
                    classes="settings-section-copy",
                )
                with Static(classes="settings-item"):
                    yield Label("Recording prefix", classes="settings-key")
                    yield Input(self.config.recording_prefix, id="setting-prefix", classes="settings-input")
                with Static(classes="settings-item"):
                    yield Label("Theme accents", classes="settings-key")
                    yield Select.from_values(
                        THEMES, value=self.config.theme, id="setting-theme", classes="settings-input"
                    )
                with Horizontal(classes="settings-toggle-row"):
                    with Static(classes="settings-toggle-copy"):
                        yield Label("Copy automatically", classes="settings-key")
                        yield Label("Put complete transcriptions on the clipboard", classes="settings-help")
                    yield Switch(value=self.config.copy_complete_text, id="setting-copy")

            with Static(classes="settings-group", id="shortcut-group"):
                yield Label("Desktop shortcut", classes="settings-section-title")
                yield Label(
                    f"{self._shortcut.hint} · {'configured' if self._shortcut.configured else 'setup required'}",
                    id="shortcut-status",
                    classes="settings-section-copy",
                )
                yield Static(self._shortcut.command, id="shortcut-command")
                with Horizontal(id="shortcut-actions"):
                    yield VoiceButton("󰆏  copy command", id="shortcut-copy-btn")
                    yield VoiceButton("󰍜  keyboard settings", id="shortcut-settings-btn")
        with Static(id="settings-footer"):
            yield Label("", id="settings-status")
            yield VoiceButton("󰉋  save changes", role="primary", id="settings-save-btn")

    def on_mount(self) -> None:
        self._control.start()
        self._ensure_desktop_status().set_state("initializing")
        self.set_interval(0.1, self._update_timer)
        self._load_history()
        self._activate()

    def on_unmount(self) -> None:
        self._control.stop()
        microphone, job = self._microphone, self._job
        if microphone is not None and job is not None:
            try:
                with self._finalization_lock:
                    if microphone.is_recording:
                        artifact, result = self.runtime.stop_recording(microphone, job)
                        persist_markdown(artifact.path, result, self.config.markdown_path)
            except Exception:
                logger.exception("TUI shutdown could not finalize the active recording")
        if self._desktop_status is not None:
            self._desktop_status.stop()
        try:
            self.runtime.close()
        except Exception:
            logger.exception("TUI shutdown could not close the runtime")

    @work(thread=True, exclusive=True, group="activation")
    def _activate(self) -> None:
        try:
            active = self.runtime.activate()
        except Exception as error:
            self.call_from_thread(self._set_error, f"activation failed: {error}")
            return
        self.call_from_thread(self._set_ready, active.device_name)

    def _set_status(self, state: str, message: str) -> None:
        icons = {"ready": "", "recording": "󰑊", "transcribing": "󰔟", "error": "󰅙"}
        label = self.query_one("#status", Label)
        label.remove_class("ready", "recording", "transcribing", "error")
        label.add_class(state)
        label.update(f"{icons.get(state, '󰔟')}  {message}")

    def _set_ready(self, device_name: str, announce: bool = True) -> None:
        del device_name
        self._state = "ready"
        self._set_status("ready", "ready")
        self.query_one("#header-model", Label).update("[dim]model:[/] Parakeet v3  [dim]device:[/] NVIDIA CUDA · FP16")
        if announce and self._desktop_status is not None:
            self._desktop_status.set_state("ready")

    def _set_error(self, message: str) -> None:
        notify = self._state == "loading" or self._external_session
        self._state = "error"
        self._set_status("error", message)
        if notify and self._desktop_status is not None:
            self._desktop_status.set_state("error")
        self._external_session = False

    @on(VoiceButton.Pressed, "#tx-copy-btn")
    def copy_pressed(self) -> None:
        self.action_copy()

    @on(VoiceButton.Pressed, "#settings-save-btn")
    def settings_pressed(self) -> None:
        self.action_save_settings()

    @on(VoiceButton.Pressed, "#shortcut-copy-btn")
    def shortcut_copy_pressed(self) -> None:
        if copy_to_clipboard(self._shortcut.command):
            self.notify("Copied desktop toggle command")

    @on(VoiceButton.Pressed, "#shortcut-settings-btn")
    def shortcut_settings_pressed(self) -> None:
        try:
            open_shortcut_settings()
        except RuntimeError as error:
            self.notify(str(error), severity="warning")

    def action_toggle_recording(self) -> None:
        self._toggle_recording(external=False)

    def _external_toggle(self) -> None:
        self._ensure_desktop_status()
        self.query_one("#tabs", TabbedContent).active = "tab-record"
        self._toggle_recording(external=True)

    def _toggle_recording(self, *, external: bool) -> None:
        if self._state == "ready":
            self._external_session = external
            self._state = "starting"
            self._set_status("transcribing", "starting recording…")
            self._overlay_set("recording")
            self._start_recording()
        elif self._state == "recording":
            self._state = "transcribing"
            self._set_status("transcribing", "finalizing WAV and transcription…")
            self._overlay_set("transcribing")
            self._stop_recording()
        elif external:
            self._external_session = True
            self._overlay_set("error")

    def _ensure_desktop_status(self) -> DesktopStatus:
        if self._desktop_status is None:
            self._desktop_status = DesktopStatus()
            self._desktop_status.start()
        return self._desktop_status

    def _overlay_set(self, state: DesktopStatusState) -> None:
        if self._external_session and self._desktop_status is not None:
            self._desktop_status.set_state(state)

    @work(thread=True, exclusive=True, group="recording-start")
    def _start_recording(self) -> None:
        try:
            microphone, job = self.runtime.start_recording(on_update=self._receive_live_update)
        except Exception as error:
            self.call_from_thread(self._set_error, f"recording failed: {error}")
            return
        self.call_from_thread(self._recording_started, microphone, job)

    def _receive_live_update(self, update: GrowingTranscriptionUpdate) -> None:
        self.call_from_thread(self._show_live_update, update)

    def _show_live_update(self, update: GrowingTranscriptionUpdate) -> None:
        if self._state not in {"recording", "transcribing"}:
            return
        text = self.query_one("#tx-text", Label)
        text.remove_class("placeholder")
        text.update(update.text or "Listening…")
        self.query_one("#tx-meta", Label).update(
            f"live · {update.processed_chunks} chunks · through {update.processed_through_sample / 16_000:.1f}s"
        )

    def _recording_started(self, microphone: MicrophoneStream, job: GrowingTranscriptionJob) -> None:
        self._microphone = microphone
        self._job = job
        self._record_started = time.monotonic()
        self._state = "recording"
        self._set_status("recording", "recording… 󰔛 0.0s")

    def _update_timer(self) -> None:
        if self._state != "recording":
            return
        elapsed = time.monotonic() - self._record_started
        if self._microphone is not None and self._microphone.capture_error is not None:
            logger.error("TUI detected capture failure after %.3fs", elapsed)
            self._state = "transcribing"
            self._set_status("transcribing", "capture failed; preserving partial audio…")
            self._stop_recording()
            return
        if self._external_session and self._desktop_status is not None:
            self._desktop_status.set_recording_elapsed(elapsed)
        minutes, seconds = divmod(int(elapsed), 60)
        rendered = f"{minutes:02d}:{seconds:02d}" if minutes else f"{elapsed:.1f}s"
        self._set_status("recording", f"recording… 󰔛 {rendered}")

    @work(thread=True, exclusive=True, group="recording-stop")
    def _stop_recording(self) -> None:
        microphone, job = self._microphone, self._job
        if microphone is None or job is None:
            self.call_from_thread(self._set_error, "recording state is incomplete")
            return
        try:
            with self._finalization_lock:
                artifact, result = self.runtime.stop_recording(microphone, job)
                markdown = persist_markdown(artifact.path, result, self.config.markdown_path)
                copied = False
                if result.complete and result.text and self.config.copy_complete_text:
                    copied = copy_to_clipboard(result.text)
        except Exception as error:
            logger.exception("Recording finalization failed")
            self.call_from_thread(self._set_error, f"transcription failed: {error}")
            return
        self.call_from_thread(self._recording_finished, markdown, result, copied)

    def _recording_finished(
        self,
        markdown_path: Path,
        result: FileTranscriptionResult,
        copied: bool,
    ) -> None:
        self._microphone = None
        self._job = None
        self._last_result = result
        text = self.query_one("#tx-text", Label)
        text.remove_class("placeholder")
        text.update(result.text or "No speech detected.")
        self.query_one("#tx-meta", Label).update(
            f"{result.duration_seconds:.1f}s · {result.latency_seconds:.1f}s processing · "
            f"{len(result.chunks)} chunks · {'complete' if result.complete else 'incomplete'}"
        )
        self.query_one("#tx-copy-btn", VoiceButton).disabled = not (result.complete and bool(result.text))
        self._load_history(select=markdown_path)
        if result.complete:
            self._overlay_set("copied" if copied else "hidden")
            self._set_ready(result.deployment.device_name, announce=False)
        else:
            self._overlay_set("error")
            self._state = "ready"
            self._set_status("error", "incomplete result saved; audio preserved")
        self._external_session = False

    def action_copy(self) -> None:
        if self._last_result is None or not self._last_result.complete or not self._last_result.text:
            return
        if copy_to_clipboard(self._last_result.text):
            self.notify("Copied transcription")
        else:
            self.notify("Clipboard is unavailable", severity="warning")

    @on(OptionList.OptionHighlighted, "#history-options")
    def history_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._show_history_entry(event.option_index)

    def _show_history_entry(self, index: int) -> None:
        if not 0 <= index < len(self._history):
            return
        entry = self._history[index]
        recorded_at = _recorded_at(entry.markdown.stem)
        title = recorded_at.strftime("%A, %B %d · %H:%M:%S") if recorded_at is not None else entry.markdown.stem
        status = "Complete" if entry.complete else "Incomplete"
        audio = "WAV available" if entry.audio is not None else "audio unavailable"
        self.query_one("#history-detail-title", Label).update(title)
        self.query_one("#history-detail-meta", Label).update(
            f"{entry.duration_seconds:.1f} seconds  ·  {status}  ·  {audio}"
        )
        self.query_one("#history-viewer", Markdown).update(entry.text or "_No speech detected._")

    def _load_history(self, select: Path | None = None) -> None:
        self._history = _read_history(self.config)
        options = self.query_one("#history-options", OptionList)
        options.clear_options()
        self.query_one("#history-list-pane", Static).border_title = f"recordings · {len(self._history)}"
        selected_index = None
        for index, entry in enumerate(self._history):
            options.add_option(_history_label(entry))
            if select == entry.markdown:
                selected_index = index
        if self._history:
            selected_index = selected_index if selected_index is not None else 0
            options.highlighted = selected_index
            self._show_history_entry(selected_index)
        else:
            self.query_one("#history-detail-title", Label).update("No recordings yet")
            self.query_one("#history-detail-meta", Label).update("Your completed dictations will appear here.")
            self.query_one("#history-viewer", Markdown).update("_Record something to begin your local history._")

    def action_save_settings(self) -> None:
        if self._state in {"recording", "starting", "transcribing"}:
            self.query_one("#settings-status", Label).update("Stop recording before saving settings.")
            return
        try:
            selected_theme = self.query_one("#setting-theme", Select).value
            if not isinstance(selected_theme, str):
                raise ValueError("Select a theme.")
            updated = AppConfig(
                deployment_id=self.config.deployment_id,
                recordings_path=Path(self.query_one("#setting-recordings", Input).value).expanduser(),
                markdown_path=Path(self.query_one("#setting-markdown", Input).value).expanduser(),
                artifact_cache_path=Path(self.query_one("#setting-artifacts", Input).value).expanduser(),
                recording_prefix=self.query_one("#setting-prefix", Input).value,
                input_device_index=self.config.input_device_index,
                copy_complete_text=self.query_one("#setting-copy", Switch).value,
                theme=selected_theme,
            )
            save_config(updated)
        except Exception as error:
            self.query_one("#settings-status", Label).update(f"Settings error: {error}")
            return
        self.runtime.close()
        self.config = updated
        self.runtime = ApplicationRuntime(updated)
        self.theme = updated.theme
        self.query_one("#settings-status", Label).update("Saved. Reloading verified deployment…")
        self._state = "loading"
        self._set_status("transcribing", "loading model…")
        if self._desktop_status is not None:
            self._desktop_status.set_state("initializing")
        self._load_history()
        self._activate()


def _recorded_at(stem: str) -> datetime | None:
    parts = stem.rsplit("_", 4)
    if len(parts) != 5:
        return None
    try:
        return datetime.strptime("".join(parts[1:4]), "%Y%m%d%H%M%S%f")
    except ValueError:
        return None


def _history_label(entry: HistoryEntry) -> str:
    recorded_at = _recorded_at(entry.markdown.stem)
    identity = recorded_at.strftime("%b %d  %H:%M:%S") if recorded_at is not None else entry.markdown.stem
    if len(identity) > 22:
        identity = f"{identity[:21]}…"
    marker = "✓" if entry.complete else "!"
    return f"{marker}  {identity}  {entry.duration_seconds:>6.1f}s"


def _read_history(config: AppConfig) -> list[HistoryEntry]:
    entries = []
    if not config.markdown_path.exists():
        return entries
    for markdown in sorted(config.markdown_path.glob("*.md"), reverse=True):
        try:
            content = markdown.read_text(encoding="utf-8")
            metadata, text = _split_markdown(content)
            audio_name = metadata.get("audio")
            audio = config.recordings_path / audio_name if audio_name else None
            entries.append(
                HistoryEntry(
                    markdown,
                    audio if audio is not None and audio.exists() else None,
                    float(metadata.get("duration_seconds", "0")),
                    metadata.get("complete") == "true",
                    text,
                )
            )
        except (OSError, ValueError):
            continue
    return entries


def _split_markdown(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---\n") or "\n---\n" not in content[4:]:
        raise ValueError("not a VoicePad schema-1 Markdown result")
    header, text = content[4:].split("\n---\n", 1)
    metadata = {}
    for line in header.splitlines():
        if ":" in line and not line.startswith("  "):
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata, text.strip()


def run(config: AppConfig | None = None) -> None:
    VoicePadApp(config).run()
