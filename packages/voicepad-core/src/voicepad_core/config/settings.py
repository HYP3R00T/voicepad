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

    # Transcription settings
    transcription_model: str = Field(
        default="tiny",
        description="Whisper model name (use 'voicepad config models' to see available models)",
    )

    transcription_device: str = Field(
        default="auto",
        description="Device for transcription (auto/cuda/cpu)",
    )

    transcription_compute_type: str = Field(
        default="auto",
        description="Compute precision (auto/float16/int8/float32)",
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
