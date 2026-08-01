from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import (
    Checkbox,
    Footer,
    Input,
    Label,
    Markdown,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from voicepad_core.audio import MicrophoneStream
from voicepad_core.pipeline import (
    FileTranscriptionResult,
    GrowingTranscriptionJob,
    GrowingTranscriptionUpdate,
)

from voicepad.config import AliasConfiguration, AppConfig, load_config, save_config
from voicepad.output import persist_markdown
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.components import VoiceButton
from voicepad.tui.control import ControlServer
from voicepad.tui.theme import THEMES
from voicepad.tui.utils.clipboard import copy_to_clipboard

try:
    APP_VERSION = f"v{version('voicepad')}"
except Exception:
    APP_VERSION = "dev"


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
        super().__init__()
        self.config = config or load_config()
        self.theme = self.config.theme
        self.runtime = runtime or ApplicationRuntime(self.config)
        self._state = "loading"
        self._microphone: MicrophoneStream | None = None
        self._job: GrowingTranscriptionJob | None = None
        self._last_result: FileTranscriptionResult | None = None
        self._record_started = 0.0
        self._history: list[HistoryEntry] = []
        self._control = ControlServer(lambda: self.call_from_thread(self.action_toggle_recording))

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
                    history_view.border_title = "transcription"
                    yield Markdown("# No recording selected", id="history-viewer")
            with TabPane("  settings  ", id="tab-settings"):
                yield from self._compose_settings()
        yield Footer()

    def _compose_settings(self) -> ComposeResult:
        with VerticalScroll(id="settings-scroll"), Static(id="settings-fields"):
            yield Label("recordings path", classes="settings-key")
            yield Input(str(self.config.recordings_path), id="setting-recordings", classes="settings-input")
            yield Label("markdown path", classes="settings-key")
            yield Input(str(self.config.markdown_path), id="setting-markdown", classes="settings-input")
            yield Label("model artifact cache", classes="settings-key")
            yield Input(str(self.config.artifact_cache_path), id="setting-artifacts", classes="settings-input")
            yield Label("recording filename prefix", classes="settings-key")
            yield Input(self.config.recording_prefix, id="setting-prefix", classes="settings-input")
            yield Label("theme", classes="settings-key")
            yield Select.from_values(THEMES, value=self.config.theme, id="setting-theme", classes="settings-input")
            yield Checkbox(
                "copy complete transcription automatically",
                value=self.config.copy_complete_text,
                id="setting-copy",
            )
            yield Label("proper-noun aliases — Canonical = alias | alias", classes="settings-key")
            yield TextArea(self._render_aliases(), id="setting-proper_nouns")
        with Static(id="settings-footer"):
            yield Label("", id="settings-status")
            yield VoiceButton("󰉋  save", role="primary", id="settings-save-btn")

    def on_mount(self) -> None:
        self._control.start()
        self.set_interval(0.1, self._update_timer)
        self._load_history()
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
            self.call_from_thread(self._set_error, f"activation failed: {error}")
            return
        self.call_from_thread(self._set_ready, active.device_name)

    def _set_status(self, state: str, message: str) -> None:
        icons = {"ready": "", "recording": "󰑊", "transcribing": "󰔟", "error": "󰅙"}
        label = self.query_one("#status", Label)
        label.remove_class("ready", "recording", "transcribing", "error")
        label.add_class(state)
        label.update(f"{icons.get(state, '󰔟')}  {message}")

    def _set_ready(self, device_name: str) -> None:
        del device_name
        self._state = "ready"
        self._set_status("ready", "ready")
        self.query_one("#header-model", Label).update("[dim]model:[/] Parakeet v3  [dim]device:[/] NVIDIA CUDA · FP16")

    def _set_error(self, message: str) -> None:
        self._state = "error"
        self._set_status("error", message)

    @on(VoiceButton.Pressed, "#tx-copy-btn")
    def copy_pressed(self) -> None:
        self.action_copy()

    @on(VoiceButton.Pressed, "#settings-save-btn")
    def settings_pressed(self) -> None:
        self.action_save_settings()

    def action_toggle_recording(self) -> None:
        if self._state == "ready":
            self._state = "starting"
            self._set_status("transcribing", "starting recording…")
            self._start_recording()
        elif self._state == "recording":
            self._state = "transcribing"
            self._set_status("transcribing", "finalizing WAV and transcription…")
            self._stop_recording()

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
            artifact, result = self.runtime.stop_recording(microphone, job)
            markdown = persist_markdown(artifact.path, result, self.config.markdown_path)
            if result.complete and result.text and self.config.copy_complete_text:
                copy_to_clipboard(result.text)
        except Exception as error:
            self.call_from_thread(self._set_error, f"transcription failed: {error}")
            return
        self.call_from_thread(self._recording_finished, markdown, result)

    def _recording_finished(self, markdown_path: Path, result: FileTranscriptionResult) -> None:
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
            self._set_ready(result.deployment.device_name)
        else:
            self._state = "ready"
            self._set_status("error", "incomplete result saved; audio preserved")

    def action_copy(self) -> None:
        if self._last_result is None or not self._last_result.complete or not self._last_result.text:
            return
        if copy_to_clipboard(self._last_result.text):
            self.notify("Copied transcription")
        else:
            self.notify("Clipboard is unavailable", severity="warning")

    @on(OptionList.OptionSelected, "#history-options")
    def history_selected(self, event: OptionList.OptionSelected) -> None:
        if 0 <= event.option_index < len(self._history):
            self.query_one("#history-viewer", Markdown).update(
                self._history[event.option_index].markdown.read_text(encoding="utf-8")
            )

    def _load_history(self, select: Path | None = None) -> None:
        self._history = _read_history(self.config)
        options = self.query_one("#history-options", OptionList)
        options.clear_options()
        selected_index = None
        for index, entry in enumerate(self._history):
            marker = "✓" if entry.complete else "!"
            options.add_option(f"{marker}  {entry.markdown.stem}  {entry.duration_seconds:.1f}s")
            if select == entry.markdown:
                selected_index = index
        if selected_index is not None:
            options.highlighted = selected_index
            self.query_one("#history-viewer", Markdown).update(
                self._history[selected_index].markdown.read_text(encoding="utf-8")
            )

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
                copy_complete_text=self.query_one("#setting-copy", Checkbox).value,
                theme=selected_theme,
                proper_nouns=_parse_aliases(self.query_one("#setting-proper_nouns", TextArea).text),
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
        self._load_history()
        self._activate()

    def _render_aliases(self) -> str:
        return "\n".join(f"{item.canonical} = {' | '.join(item.aliases)}" for item in self.config.proper_nouns)


def _parse_aliases(value: str) -> tuple[AliasConfiguration, ...]:
    parsed = []
    for line_number, line in enumerate(value.splitlines(), start=1):
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"Alias line {line_number} must contain '='.")
        canonical, alternatives = line.split("=", 1)
        aliases = tuple(alias.strip() for alias in alternatives.split("|") if alias.strip())
        parsed.append(AliasConfiguration(canonical.strip(), aliases))
    return tuple(parsed)


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
