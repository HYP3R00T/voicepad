"""Setup wizard modal for VoicePad TUI."""

from __future__ import annotations

import contextlib
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, ClassVar

from textual import on, work
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, ProgressBar, Select, Static
from voicepad_core import VALID_TRANSCRIPTION_MODELS

from voicepad.tui.components import VoiceButton

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from voicepad_core.config import Config

try:
    _APP_VERSION = f"v{_pkg_version('voicepad')}"
except Exception:
    _APP_VERSION = "dev"

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


class SetupModal(ModalScreen[tuple[str, int | None]]):
    """4-step first-run wizard.

    Step 0 — Welcome
    Step 1 — Model selection + download
    Step 2 — Microphone selection
    Step 3 — Finish / config info
    """

    BINDINGS: ClassVar[list[Binding]] = []

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._step: int = 0
        self._chosen_model: str = config.transcription_model
        self._chosen_device_index: int | None = config.input_device_index
        self._downloading = False
        self._download_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Static(id="wizard-dialog"):
            yield Static("", id="wizard-step-indicator")
            yield Static(id="wizard-body")
            with Static(id="wizard-nav"):
                yield VoiceButton("← Back", role="default", id="wizard-back")
                yield Static("", id="wizard-spacer")
                yield VoiceButton("Next →", role="primary", id="wizard-next")

    def on_mount(self) -> None:
        self._render_step()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#wizard-next")
    def on_next(self) -> None:
        if self._step == 1:
            if self._downloading:
                return  # download in progress — button is disabled anyway
            next_btn = self.query_one("#wizard-next", VoiceButton)
            if str(next_btn.label) in ("Download →", "Download & Continue →"):
                # First click on step 1 — start the download
                self._start_download()
                return
            # Button now says "Continue →" — download finished, advance
        if self._step < 3:
            self._step += 1
            self._render_step()
        else:
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
        for child in list(body.children):
            child.remove()

        back_btn = self.query_one("#wizard-back", VoiceButton)
        next_btn = self.query_one("#wizard-next", VoiceButton)
        back_btn.display = self._step > 0
        next_btn.label = "Finish" if self._step == 3 else "Download →" if self._step == 1 else "Next →"
        next_btn.disabled = False

        if self._step == 0:
            self._mount_step_welcome(body)
        elif self._step == 1:
            self._mount_step_model(body)
        elif self._step == 2:
            self._mount_step_microphone(body)
        elif self._step == 3:
            self._mount_step_finish(body)

        self.set_timer(0.05, lambda: next_btn.focus())

    # ------------------------------------------------------------------
    # Step 0 — Welcome
    # ------------------------------------------------------------------

    def _mount_step_welcome(self, body: Static) -> None:
        body.mount(Static(f"󰍬  VoicePad  {_APP_VERSION}", classes="wizard-title"))
        body.mount(Static("Your private, local-first dictation studio.", classes="wizard-text"))
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
        # Indeterminate progress bar — no percentage, just a looping animation
        bar = ProgressBar(id="wizard-progress", show_eta=False, show_percentage=False)
        bar.display = False
        body.mount(bar)

    @on(Select.Changed, "#wizard-model-select")
    def on_model_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._chosen_model = str(event.value)
            with contextlib.suppress(Exception):
                self.query_one("#wizard-model-hint", Static).update(_HINTS.get(self._chosen_model, ""))

    def _start_download(self) -> None:
        if self._downloading:
            return
        self._downloading = True
        self._download_start_time = 0.0
        self.query_one("#wizard-next", VoiceButton).disabled = True
        self._download_model_worker(self._chosen_model)

    def _start_elapsed_timer(self) -> None:
        """Tick the elapsed-time display every second while downloading."""
        import time

        if not self._downloading:
            return
        elapsed = time.monotonic() - self._download_start_time
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        with contextlib.suppress(Exception):
            self._set_download_status(f"Downloading {self._chosen_model}  ·  {elapsed_str}")
        # Reschedule for next second
        self.set_timer(1.0, self._start_elapsed_timer)

    @work(thread=True, name="setup-download")
    def _download_model_worker(self, model: str) -> None:
        import time

        from voicepad_core import TranscriptionError, ensure_model_downloaded, model_downloaded

        self.app.call_from_thread(self._set_download_status, f"Checking {model}")

        if model_downloaded(model):
            self.app.call_from_thread(self._on_download_done, model, None, 0.0)
            return

        # Record start time and kick off the indeterminate bar + elapsed timer
        self._download_start_time = time.monotonic()
        self.app.call_from_thread(self._show_progress_bar)
        self.app.call_from_thread(self._set_download_status, f"Downloading {model}")
        self.app.call_from_thread(self.set_timer, 1.0, self._start_elapsed_timer)

        try:
            ensure_model_downloaded(model, on_progress=None)
            elapsed = time.monotonic() - self._download_start_time
            self.app.call_from_thread(self._on_download_done, model, None, elapsed)
        except TranscriptionError as e:
            self.app.call_from_thread(self._on_download_done, model, str(e), 0.0)

    def _show_progress_bar(self) -> None:
        with contextlib.suppress(Exception):
            bar = self.query_one("#wizard-progress", ProgressBar)
            bar.display = True
            # total=None makes Textual render an indeterminate looping bar
            bar.update(total=None)

    def _set_download_status(self, msg: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#wizard-download-status", Static).update(msg)

    def _on_download_done(self, model: str, error: str | None, elapsed: float) -> None:
        self._downloading = False
        with contextlib.suppress(Exception):
            self.query_one("#wizard-progress", ProgressBar).display = False
        if error:
            self._set_download_status(f"[red]\U000f0156  Download failed: {error}[/]")
            # Re-enable the Download button so the user can retry
            next_btn = self.query_one("#wizard-next", VoiceButton)
            next_btn.label = "Download →"
            next_btn.disabled = False
        else:
            # Show completion message with total time, then unlock Continue
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            self._set_download_status(f"[green]\U000f012c  Downloaded {model}[/]  [dim]·  took {time_str}[/]")
            next_btn = self.query_one("#wizard-next", VoiceButton)
            next_btn.label = "Continue →"
            next_btn.disabled = False

    # ------------------------------------------------------------------
    # Step 2 — Microphone
    # ------------------------------------------------------------------

    def _mount_step_microphone(self, body: Static) -> None:
        from voicepad.cli.config import _get_input_devices

        body.mount(Static("Select Your Microphone", classes="wizard-title"))
        body.mount(Static("[dim]System default[/] works for most setups.", classes="wizard-text"))

        devices = _get_input_devices()
        device_options: list[tuple[str, int]] = [("System default", -1)]
        device_options += [(d.name, d.index) for d in devices]

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
        body.mount(Static(f"[dim]{len(devices)} input device(s) found[/]", classes="wizard-hint"))

    @on(Select.Changed, "#wizard-device-select")
    def on_device_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            v = int(str(event.value))
            self._chosen_device_index = None if v == -1 else v

    # ------------------------------------------------------------------
    # Step 3 — Finish
    # ------------------------------------------------------------------

    def _mount_step_finish(self, body: Static) -> None:
        from utilityhub_config import get_config_path

        config_path = get_config_path("voicepad", format="yaml")
        body.mount(Static("󰄬  You're all set!", classes="wizard-title"))
        body.mount(Static("VoicePad is ready. Here are the essentials:", classes="wizard-text"))

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
