"""TUI-specific configuration for VoicePad.

Kept separate from voicepad-core so the core package stays UI-agnostic.
Stored at ~/.config/voicepad/voicepad-ui.yaml alongside the main config.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from utilityhub_config import get_config_path, write_config

_UI_CONFIG_PATH = get_config_path("voicepad", format="yaml").parent / "voicepad-ui.yaml"

_DEFAULT_THEME = "tokyo-night"


class TUIConfig(BaseModel):
    """UI-layer settings that have no bearing on core transcription logic."""

    model_config = ConfigDict(frozen=True)

    theme: str = Field(
        default=_DEFAULT_THEME,
        description=(
            "UI theme for the application. "
            "Available Textual themes: atom-one-dark, atom-one-light, catppuccin-frappe, "
            "catppuccin-latte, catppuccin-macchiato, catppuccin-mocha, dracula, flexoki, "
            "gruvbox, monokai, nord, rose-pine, rose-pine-dawn, rose-pine-moon, "
            "solarized-dark, solarized-light, textual-dark, textual-light, tokyo-night."
        ),
    )


def load_tui_config() -> TUIConfig:
    """Load TUI config from disk, returning defaults if the file doesn't exist."""
    if not _UI_CONFIG_PATH.exists():
        return TUIConfig()
    try:
        import yaml

        raw = yaml.safe_load(_UI_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return TUIConfig(**raw)
    except Exception:
        return TUIConfig()


def save_tui_config(config: TUIConfig) -> None:
    """Persist TUI config to disk."""
    _UI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_config(config, "voicepad", path=_UI_CONFIG_PATH, format="yaml")
