"""Core configuration for voicepad."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from utilityhub_config import expand_path, load_settings


def _get_available_models() -> tuple[str, ...]:
    """Query available models from faster_whisper at import time."""
    try:
        from faster_whisper.utils import available_models

        return tuple(available_models())
    except Exception:
        return (
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
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


VALID_TRANSCRIPTION_MODELS: tuple[str, ...] = _get_available_models()


class Config(BaseModel):
    """Voicepad configuration with hierarchical loading.

    Configuration sources (highest to lowest precedence):
    1. Environment variables (VOICEPAD_*)
    2. ./voicepad.yaml (current directory)
    3. ~/.config/voicepad/voicepad.yaml (user config)
    4. Default values
    """

    model_config = ConfigDict(frozen=True)

    recordings_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/data/recordings"),
        description="Directory where WAV recordings are saved.",
    )
    markdown_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/data/markdown"),
        description="Directory where markdown transcriptions are saved.",
    )
    model_cache_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/models"),
        description="Directory where Whisper model weights are cached.",
    )
    vad_model_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/models/vad"),
        description="Directory where VAD (Voice Activity Detection) model files are stored.",
    )
    logs_path: Path = Field(
        default_factory=lambda: expand_path("~/.config/voicepad/logs"),
        description="Directory where transcription logs are saved.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level for transcription operations.",
    )

    input_device_index: int | None = Field(
        default=None,
        description="Audio input device index. None = system default. "
        "Run 'voicepad config input list' to see available devices.",
    )
    recording_prefix: str = Field(
        default="recording",
        description="Filename prefix for recordings (timestamp is appended automatically).",
    )

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
        """Validate model name against faster_whisper's available models."""
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

    global_hotkey: str = Field(
        default="<ctrl>+<alt>+v",
        description=(
            "System-wide hotkey to start/stop recording from any application. "
            "Press once to start, press again to stop and copy transcription to clipboard. "
            "Uses pynput key syntax, e.g. '<ctrl>+<alt>+v' or '<ctrl>+<shift>+space'. "
            "Set to empty string to disable."
        ),
    )

    language: str = Field(
        default="en",
        description=(
            "Primary language for transcription. Non-English languages are supported "
            "but may have reduced accuracy. A warning is emitted for non-English use."
        ),
    )

    silence_threshold_ms: int = Field(
        default=1000,
        description=(
            "VAD silence duration (ms) to trigger a chunk split during streaming. "
            "Benchmarkable — test with 500, 800, 1000, 1500 to find optimal value."
        ),
    )

    min_chunk_s: float = Field(
        default=15.0,
        description=(
            "Minimum audio duration (seconds) before considering a silence-triggered split. "
            "Benchmarkable — test with 10, 15, 20, 29."
        ),
    )

    max_chunk_s: float = Field(
        default=29.0,
        description=(
            "Per-chunk safety limit in seconds. Long recording sessions remain unbounded; "
            "this only forces a split before Whisper's 30s context window is exceeded."
        ),
    )

    overlap_s: float = Field(
        default=0.5,
        description="Audio overlap (seconds) kept at chunk boundaries for acoustic continuity.",
    )

    local_agreement_mic: bool = Field(
        default=False,
        description=(
            "Enable two-pass LocalAgreement verification for mic streaming mode. "
            "Roughly doubles per-chunk latency but improves accuracy."
        ),
    )

    local_agreement_file: bool = Field(
        default=True,
        description=(
            "Enable two-pass LocalAgreement verification for file retranscription. "
            "No latency impact — user is already waiting for full-file processing."
        ),
    )

    @field_validator(
        "recordings_path", "markdown_path", "model_cache_path", "vad_model_path", "logs_path", mode="before"
    )
    @classmethod
    def expand_paths(cls, v: Path | str) -> Path:
        """Expand ~ and environment variables in path fields."""
        return expand_path(str(v) if isinstance(v, Path) else v)

    @model_validator(mode="after")
    def ensure_paths_expanded(self) -> Config:
        """Expand all path fields (safety check for default values)."""
        for path_field in ("recordings_path", "markdown_path", "model_cache_path", "vad_model_path", "logs_path"):
            current_path = getattr(self, path_field)
            expanded = expand_path(str(current_path))
            object.__setattr__(self, path_field, expanded)
        return self


def get_config(cwd: Path | None = None, app_name: str = "voicepad") -> Config:
    """Load configuration with hierarchical precedence.

    Args:
        cwd: Optional working directory for config search
        app_name: Application name for config file lookup

    Returns:
        Loaded configuration object
    """
    settings, _ = load_settings(Config, app_name=app_name, cwd=cwd)
    return settings


def get_config_with_metadata(
    cwd: Path | None = None,
    app_name: str = "voicepad",
) -> tuple[Config, Any]:
    """Load configuration with source metadata.

    Args:
        cwd: Optional working directory for config search
        app_name: Application name for config file lookup

    Returns:
        Tuple of (config, metadata) where metadata tracks source of each field
    """
    return load_settings(Config, app_name=app_name, cwd=cwd)
