"""Configuration management for Voicepad."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from utilityhub_config import load_settings

SUPPORTED_MODEL_SIZES = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "distil-small.en",
    "medium",
    "medium.en",
    "distil-medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-large-v3",
    "large-v3-turbo",
    "turbo",
    "auto",
)

DeviceType = Literal["cuda", "cpu", "auto"]
ModelSize = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "distil-small.en",
    "medium",
    "medium.en",
    "distil-medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-large-v3",
    "large-v3-turbo",
    "turbo",
    "auto",
]
ComputeType = Literal["float16", "int8", "int8_float16", "auto"]


class TranscriptionConfig(BaseModel):
    """Configuration for audio transcription."""

    model: ModelSize = Field(
        default="auto",
        description="Whisper model size to use for transcription",
    )
    device: DeviceType = Field(
        default="auto",
        description="Device to use for transcription (cuda, cpu, or auto-detect)",
    )
    compute_type: ComputeType = Field(
        default="auto",
        description="Compute type for transcription (float16, int8, or auto-detect)",
    )
    language: str | None = Field(
        default=None,
        description="Default language for transcription (None for auto-detection)",
    )


class Config(BaseModel):
    """Configuration for Voicepad."""

    recordings_path: Path = Path("data/recordings")
    markdown_path: Path = Path("data/markdown")
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)


def get_config() -> Config:
    """Load and return configuration from voicepad.yaml, environment, or defaults."""
    config, _ = load_settings(Config, app_name="voicepad")
    return config
