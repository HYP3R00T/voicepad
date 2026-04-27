"""Core configuration for voicepad."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from utilityhub_config import expand_path, load_settings


class Config(BaseModel):
    """Voicepad configuration.

    Loaded from voicepad.yaml with precedence:
    env vars (VOICEPAD_*) > ./voicepad.yaml > ~/.config/voicepad/voicepad.yaml > defaults
    """

    model_config = ConfigDict(frozen=True)

    recordings_path: Path = Field(
        default=Path("data/recordings"),
        description="Directory where WAV recordings are saved",
    )
    markdown_path: Path = Field(
        default=Path("data/markdown"),
        description="Directory where markdown transcriptions are saved",
    )
    input_device_index: int | None = Field(
        default=None,
        description="Audio input device index (None = system default)",
    )
    recording_prefix: str = Field(
        default="recording",
        description="Filename prefix for recordings",
    )
    transcription_model: str = Field(
        default="small",
        description="Whisper model: tiny, base, small, medium, large-v3, turbo",
    )

    @field_validator("recordings_path", "markdown_path", mode="before")
    @classmethod
    def expand_paths(cls, v: Path | str) -> Path:
        """Expand ~ and environment variables in path fields."""
        return expand_path(str(v) if isinstance(v, Path) else v)


def get_config(cwd: Path | None = None, app_name: str = "voicepad") -> Config:
    """Load configuration using utilityhub_config precedence rules."""
    settings, _ = load_settings(Config, app_name=app_name, cwd=cwd)
    return settings


def get_config_with_metadata(
    cwd: Path | None = None,
    app_name: str = "voicepad",
) -> tuple[Config, Any]:
    """Load configuration and return source metadata for each field."""
    return load_settings(Config, app_name=app_name, cwd=cwd)
