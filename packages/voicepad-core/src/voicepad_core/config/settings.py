from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from utilityhub_config import load_settings


class Config(BaseModel):
    """Core configuration for voicepad."""

    model_config = ConfigDict(frozen=True)

    recordings_path: Path = Path("data/recordings")
    markdown_path: Path = Path("data/markdown")

    input_device_index: int | None = Field(
        default=None,
        description="Default OS audio input device index",
    )

    recording_prefix: str = Field(
        default="recording",
        description="Prefix for audio recording filenames",
    )


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
