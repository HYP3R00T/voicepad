from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, Static

# Prefer loading settings through utilityhub_config which resolves multiple sources
from utilityhub_config.api import load_settings
from utilityhub_config.errors import ConfigValidationError

from voicepad.config import Config


def load_config(*, cwd: Path | None = None, app_name: str = "voicepad") -> Config:
    """Load `Config` using utilityhub_config.load_settings.

    Args:
        cwd: Working directory to search for project configs. If None, defaults to current working dir.
        app_name: Application name used for config lookup (defaults to 'voicepad').

    Raises:
        FileNotFoundError: If no project config exists and defaults are missing.
        ValueError: If validation fails.
    """
    try:
        cwd_arg = None if cwd is None else Path(cwd)
        cfg, metadata = load_settings(Config, app_name=app_name, cwd=cwd_arg)
        return cfg
    except ConfigValidationError as exc:  # pragma: no cover - library behavior
        raise ValueError("Invalid configuration") from exc


def save_config(config: Config, *, cwd: Path | None = None, app_name: str = "voicepad") -> Path:
    """Save `Config` to a project config YAML file.

    Writes to `cwd / {app_name}.yaml` or the current working directory when `cwd` is None.
    Returns the path written to.
    """
    target_dir = Path(cwd) if cwd is not None else Path.cwd()
    target = target_dir / f"{app_name}.yaml"
    target.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False), encoding="utf-8")
    return target


class ConfigSaved(Message):
    """Message sent when configuration was saved."""


class ConfigTab(Static):
    """A modular configuration editor panel.

    - Loads values using `utilityhub_config` (searches project config files)
    - Allows editing supported fields (recordings_path, markdown_path)
    - Saves changes back to disk (writes project `{app_name}.yaml`) and emits `ConfigSaved`

    The widget is intentionally small and focused so it can be reused/embedded
    in other UI contexts.
    """

    BINDINGS = [Binding("ctrl+s", "save", "Save config")]

    def __init__(self, *, cwd: Path | str | None = None, app_name: str = "voicepad", **kwargs: Any) -> None:
        """Create a ConfigTab.

        Args:
            cwd: Optional working directory for config lookup and saving. If a path to a file is
                 provided (e.g., 'voicepad.yaml') the parent directory will be used as the cwd.
            app_name: Application name used by utilityhub_config for file naming.
        """
        super().__init__(**kwargs)
        self.cwd: Path | None = None if cwd is None else (Path(cwd).parent if Path(cwd).suffix else Path(cwd))
        self.app_name = app_name
        self.config: Config | None = None

        # Input widgets
        self.recordings_input: Input | None = None
        self.markdown_input: Input | None = None
        self.status_label: Label | None = None

    def compose(self) -> Vertical:  # type: ignore[override]
        with Vertical():
            yield Label("Configuration", id="config-title")
            yield Label("Recordings path:")
            self.recordings_input = Input(placeholder="path to recordings", id="recordings-input")
            yield self.recordings_input
            yield Label("Markdown path:")
            self.markdown_input = Input(placeholder="path to markdown", id="markdown-input")
            yield self.markdown_input
            yield Button("Save", id="save-btn")
            self.status_label = Label("", id="config-status")
            yield self.status_label

    def on_mount(self) -> None:  # type: ignore[override]
        try:
            self.config = load_config(cwd=self.cwd, app_name=self.app_name)
            # Populate inputs
            assert self.recordings_input is not None
            assert self.markdown_input is not None
            self.recordings_input.value = str(self.config.recordings_path)
            self.markdown_input.value = str(self.config.markdown_path)
            self.set_status("Loaded configuration", success=True)
        except Exception as exc:
            self.set_status(f"Failed to load: {exc}", success=False)

    def on_button_pressed(self, _event: Button.Pressed) -> None:  # type: ignore[override]
        if _event.button.id == "save-btn":
            self.action_save()

    def action_save(self) -> None:
        """Save current input values to disk and emit `ConfigSaved`."""
        assert self.recordings_input is not None
        assert self.markdown_input is not None
        try:
            new_config = Config(
                recordings_path=Path(self.recordings_input.value.strip()),
                markdown_path=Path(self.markdown_input.value.strip()),
            )
            save_config(new_config, cwd=self.cwd, app_name=self.app_name)
            self.config = new_config
            self.set_status("Saved configuration", success=True)
            self.post_message(ConfigSaved())
        except Exception as exc:
            self.set_status(f"Save failed: {exc}", success=False)

    def set_status(self, message: str, success: bool = True) -> None:
        if self.status_label is not None:
            text = f"✅ {message}" if success else f"⚠️ {message}"
            self.status_label.update(text)


__all__ = ["ConfigTab", "load_config", "save_config"]
