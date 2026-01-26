"""Configuration management for Voicepad."""

from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field
from utilityhub_config import load_settings


class ModelInfo(NamedTuple):
    """Model metadata with category, name, and VRAM requirement."""

    category: str
    name: str
    vram: str


# Single source of truth for all model information
MODEL_INFO = (
    ModelInfo("Tiny (Fastest, Lower Accuracy)", "tiny", "~1 GB"),
    ModelInfo("Tiny (Fastest, Lower Accuracy)", "tiny.en", "~1 GB"),
    ModelInfo("Small (Balanced)", "base", "~1.5 GB"),
    ModelInfo("Small (Balanced)", "base.en", "~1.5 GB"),
    ModelInfo("Small (Balanced)", "small", "~2 GB"),
    ModelInfo("Small (Balanced)", "small.en", "~2 GB"),
    ModelInfo("Small (Balanced)", "distil-small.en", "~1.5 GB"),
    ModelInfo("Medium (Accuracy Focus)", "medium", "~3.5 GB"),
    ModelInfo("Medium (Accuracy Focus)", "medium.en", "~3.5 GB"),
    ModelInfo("Medium (Accuracy Focus)", "distil-medium.en", "~2.5 GB"),
    ModelInfo("Large (Best Quality)", "large-v1", "~4.5 GB"),
    ModelInfo("Large (Best Quality)", "large-v2", "~4.5 GB"),
    ModelInfo("Large (Best Quality)", "large-v3", "~4.5 GB"),
    ModelInfo("Large (Best Quality)", "large", "~4.5 GB"),
    ModelInfo("Large (Best Quality)", "distil-large-v2", "~2 GB"),
    ModelInfo("Large (Best Quality)", "distil-large-v3", "~2.5 GB"),
    ModelInfo("Large (Best Quality)", "large-v3-turbo", "~3 GB"),
    ModelInfo("Large (Best Quality)", "turbo", "~3 GB"),
)

# Derived exports from MODEL_INFO
MODEL_CATEGORIES = {}
for info in MODEL_INFO:
    if info.category not in MODEL_CATEGORIES:
        MODEL_CATEGORIES[info.category] = []
    MODEL_CATEGORIES[info.category].append(info.name)

VRAM_ESTIMATES = {info.name: info.vram for info in MODEL_INFO}

_MODEL_NAMES = tuple(info.name for info in MODEL_INFO)
SUPPORTED_MODEL_SIZES = _MODEL_NAMES + ("auto",)

# Generate ModelSize from SUPPORTED_MODEL_SIZES (derived from MODEL_INFO)
ModelSize = Literal[*SUPPORTED_MODEL_SIZES]  # type: ignore[misc]


class TranscriptionConfig(BaseModel):
    """Configuration for audio transcription."""

    model: ModelSize = Field(
        default="auto",
        description="Whisper model size to use for transcription",
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
