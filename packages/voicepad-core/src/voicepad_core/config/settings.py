"""Core configuration for voicepad."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from utilityhub_config import expand_path, load_settings

# ---------------------------------------------------------------------------
# Valid model names — sourced from faster_whisper.utils.available_models()
# ---------------------------------------------------------------------------

VALID_TRANSCRIPTION_MODELS: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
    "distil-large-v3",
    "distil-large-v3.5",
    "large-v3-turbo",
    "turbo",
)


class Config(BaseModel):
    """Voicepad configuration.

    Loaded from voicepad.yaml with precedence:
    env vars (VOICEPAD_*) > ./voicepad.yaml > ~/.config/voicepad/voicepad.yaml > defaults
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    recordings_path: Path = Field(
        default=Path("~/.config/voicepad/data/recordings"),
        description="Directory where WAV recordings are saved.",
    )
    markdown_path: Path = Field(
        default=Path("~/.config/voicepad/data/markdown"),
        description="Directory where markdown transcriptions are saved.",
    )

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    input_device_index: int | None = Field(
        default=None,
        description="Audio input device index. None = system default. "
        "Run 'voicepad config input list' to see available devices.",
    )
    recording_prefix: str = Field(
        default="recording",
        description="Filename prefix for recordings (timestamp is appended automatically).",
    )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    transcription_model: str = Field(
        default="turbo",
        description=(
            "Whisper model for transcription. Must be one of the supported faster-whisper models. "
            "Models are auto-downloaded from HuggingFace on first use. "
            f"Valid values: {', '.join(VALID_TRANSCRIPTION_MODELS)}."
        ),
    )

    @field_validator("transcription_model")
    @classmethod
    def validate_transcription_model(cls, v: str) -> str:
        """Reject unknown model names before any download or inference attempt."""
        if v not in VALID_TRANSCRIPTION_MODELS:
            valid = ", ".join(VALID_TRANSCRIPTION_MODELS)
            raise ValueError(f"Unknown transcription model '{v}'. Valid models: {valid}.")
        return v

    transcription_device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Device for transcription. "
            "'auto' tries CUDA first and falls back to CPU. "
            "'cuda' forces GPU (fails if unavailable). "
            "'cpu' forces CPU."
        ),
    )
    transcription_compute_type: Literal["auto", "float16", "int8", "float32", "int8_float16"] = Field(
        default="auto",
        description=(
            "Compute precision for CTranslate2. "
            "'auto' uses int8 on CUDA, int8 on CPU. "
            "'float16' — higher accuracy, ~2x VRAM vs int8 (GPU only). "
            "'int8' — best speed/VRAM tradeoff (recommended). "
            "'float32' — full precision, slowest. "
            "'int8_float16' — mixed precision."
        ),
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
