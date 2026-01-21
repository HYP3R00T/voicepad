from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Input, Label, Static

from voicepad.config import Config, get_config, save_config


class ConfigSaved(Message):
    """Message sent when configuration was saved."""


class ConfigTab(Container):
    """Configuration editor with card-based layout."""

    DEFAULT_CSS = """
    ConfigTab {
        layout: vertical;
        padding: 1;
    }

    /* Path cards row */
    #paths_row {
        height: auto;
        margin-bottom: 1;
    }

    #recordings_card {
        width: 1fr;
        height: auto;
        border: round $primary;
        padding: 0 1;
        margin-right: 1;
    }

    #markdown_card {
        width: 1fr;
        height: auto;
        border: round $primary;
        padding: 0 1;
    }

    #recordings_card Input, #markdown_card Input {
        width: 100%;
        height: 1;
        border: none;
        background: transparent;
    }

    /* Actions row */
    #actions_row {
        height: auto;
    }

    #save_card {
        width: auto;
        min-width: 10;
        height: auto;
        border: round $primary;
        padding: 0;
    }

    #save_btn {
        width: 100%;
        border: none;
        background: transparent;
    }

    #save_btn:hover {
        background: $boost;
    }

    #status_card {
        width: 1fr;
        height: auto;
        border: round $primary;
        padding: 0 1;
        margin-left: 1;
    }

    #status_row {
        height: 1;
    }

    .status-dot {
        width: 2;
    }

    .dot-ok {
        color: $success;
    }

    .dot-error {
        color: $error;
    }
    """

    BINDINGS = [Binding("ctrl+s", "save", "Save")]

    def __init__(self, *, cwd: Path | str | None = None, app_name: str = "voicepad", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cwd: Path | None = None if cwd is None else (Path(cwd).parent if Path(cwd).suffix else Path(cwd))
        self.app_name = app_name
        self.config: Config | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="paths_row"):
            recordings_card = Container(id="recordings_card")
            recordings_card.border_title = "Recordings Path"
            with recordings_card:
                yield Input(placeholder="path/to/recordings", id="recordings_input")

            markdown_card = Container(id="markdown_card")
            markdown_card.border_title = "Markdown Path"
            with markdown_card:
                yield Input(placeholder="path/to/markdown", id="markdown_input")

        with Horizontal(id="actions_row"):
            save_card = Container(id="save_card")
            save_card.border_title = "Action"
            with save_card:
                yield Button("● Save", id="save_btn")

            status_card = Container(id="status_card")
            status_card.border_title = "Status"
            with status_card, Horizontal(id="status_row"):
                yield Static("●", id="status_dot", classes="status-dot dot-ok")
                yield Label("Ready", id="status_text")

    def on_mount(self) -> None:
        try:
            self.config = get_config(app_name=self.app_name)
            recordings_input = self.query_one("#recordings_input", Input)
            markdown_input = self.query_one("#markdown_input", Input)
            recordings_input.value = str(self.config.recordings_path)
            markdown_input.value = str(self.config.markdown_path)
            self._set_status("Loaded", ok=True)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", ok=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_btn":
            self.action_save()

    def action_save(self) -> None:
        """Save current input values to disk."""
        try:
            recordings_input = self.query_one("#recordings_input", Input)
            markdown_input = self.query_one("#markdown_input", Input)
            new_config = Config(
                recordings_path=Path(recordings_input.value.strip()),
                markdown_path=Path(markdown_input.value.strip()),
            )
            save_config(new_config, app_name=self.app_name)
            self.config = new_config
            self._set_status("Saved", ok=True)
            self.post_message(ConfigSaved())
        except Exception as exc:
            self._set_status(f"Failed: {exc}", ok=False)

    def _set_status(self, message: str, ok: bool = True) -> None:
        dot = self.query_one("#status_dot", Static)
        text = self.query_one("#status_text", Label)
        if ok:
            dot.remove_class("dot-error")
            dot.add_class("dot-ok")
        else:
            dot.remove_class("dot-ok")
            dot.add_class("dot-error")
        text.update(message)


__all__ = ["ConfigTab"]
