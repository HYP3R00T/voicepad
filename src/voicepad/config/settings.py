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


def get_global_config_path(app_name: str = "voicepad") -> Path:
    """Return the global config file path: ~/.config/{app_name}/{app_name}.yaml"""
    return Path.home() / ".config" / app_name / f"{app_name}.yaml"


def _config_to_dict(config: Config) -> dict:
    """Convert Config to a dictionary suitable for YAML serialization."""
    return {
        "recordings_path": str(config.recordings_path),
        "markdown_path": str(config.markdown_path),
        "transcription": {
            "model": config.transcription.model,
            "device": config.transcription.device,
            "compute_type": config.transcription.compute_type,
            "language": config.transcription.language,
        },
    }


def ensure_config(app_name: str = "voicepad") -> Path:
    """Ensure the global config file exists with default values.

    Creates ~/.config/{app_name}/{app_name}.yaml if it doesn't exist.
    If the file exists but is empty, populates it with defaults.

    Returns:
        Path to the config file.
    """
    import yaml

    config_path = get_global_config_path(app_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create with defaults if missing or empty
    if not config_path.exists() or config_path.stat().st_size == 0:
        defaults = Config()
        config_path.write_text(
            yaml.safe_dump(_config_to_dict(defaults), sort_keys=False),
            encoding="utf-8",
        )

    return config_path


def save_config(config: Config, app_name: str = "voicepad") -> Path:
    """Save Config to the global config file.

    Saves to: ~/.config/{app_name}/{app_name}.yaml
    """
    import yaml

    config_path = get_global_config_path(app_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(_config_to_dict(config), sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def get_config(app_name: str = "voicepad", cwd: Path | None = None) -> Config:
    """Load configuration, ensuring the global config file exists first.

    Precedence (lowest to highest):
    1. Defaults from Config model
    2. Global config: ~/.config/voicepad/voicepad.yaml
    3. Project config: {cwd}/voicepad.yaml or {cwd}/config/voicepad.yaml
    4. Environment variables

    Args:
        app_name: Application name for config lookup.
        cwd: Optional working directory for project-level config discovery.
             When None, only global config is used (published package behavior).
    """
    ensure_config(app_name)
    config, _ = load_settings(Config, app_name=app_name, cwd=cwd)
    return config
