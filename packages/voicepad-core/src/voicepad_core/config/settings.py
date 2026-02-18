from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from utilityhub_config import expand_path, load_settings


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

    # VAD chunking settings (using faster-whisper's Silero VAD)
    vad_enabled: bool = Field(
        default=False,
        description="Enable Voice Activity Detection for smart chunking during recording",
    )

    vad_min_chunk_duration: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Minimum duration (seconds) before allowing chunk splits",
    )

    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Silero VAD speech probability threshold (0.0-1.0, higher = more strict)",
    )

    vad_min_silence_duration_ms: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Minimum silence duration (ms) required to trigger chunk boundary",
    )

    vad_speech_pad_ms: int = Field(
        default=400,
        ge=0,
        le=2000,
        description="Padding (ms) added to each side of detected speech segments",
    )

    @field_validator("recordings_path", "markdown_path", mode="before")
    @classmethod
    def expand_paths(cls, v: Path | str) -> Path:
        """Expand user home directory (~) and environment variables in path fields."""
        if isinstance(v, Path):
            v = str(v)
        return expand_path(v)


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
