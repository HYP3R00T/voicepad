from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from utilityhub_config import load_settings

if TYPE_CHECKING:
    from typing import Any


class Config(BaseModel):
    """Core configuration for voicepad."""

    model_config = ConfigDict(frozen=True)

    recordings_path: Path = Path("data/recordings")
    markdown_path: Path = Path("data/markdown")


def get_config(cwd: Path | None = None, app_name: str = "voicepad") -> Config:
    """Load configuration using utilityhub_config precedence rules."""
    settings, _metadata = load_settings(Config, app_name=app_name, cwd=cwd)
    return settings


def get_config_with_metadata(
    cwd: Path | None = None,
    app_name: str = "voicepad",
) -> tuple[Config, Any]:
    """Load configuration and return metadata for source tracking."""
    settings, metadata = load_settings(Config, app_name=app_name, cwd=cwd)
    return settings, metadata
