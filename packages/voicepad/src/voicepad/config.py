from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)
from utilityhub_config import expand_path, get_config_path, load_settings, write_config
from utilityhub_config.errors import ConfigError
from voicepad_core.deployments import PARAKEET_V3_CUDA

from voicepad.tui.theme import DEFAULT_THEME, THEMES

APP_NAME = "voicepad"
logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when VoicePad configuration cannot be loaded or saved."""


class AppConfig(BaseModel):
    """Validated application-owned settings consumed explicitly by VoicePad Core."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deployment_id: str = PARAKEET_V3_CUDA.id
    recordings_path: Path = Path.home() / ".config/voicepad/data/recordings"
    markdown_path: Path = Path.home() / ".config/voicepad/data/markdown"
    artifact_cache_path: Path = Path.home() / ".cache/voicepad/artifacts"
    recording_prefix: str = "recording"
    input_device_index: int | None = None
    copy_complete_text: bool = True
    theme: str = DEFAULT_THEME

    @model_serializer(mode="wrap")
    def serialize_config(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        serialized = cast(dict[str, Any], handler(self))
        return {key: value for key, value in serialized.items() if value is not None}

    @field_validator("recordings_path", "markdown_path", "artifact_cache_path", mode="before")
    @classmethod
    def expand_configured_path(cls, value: Path | str) -> Path:
        return expand_path(str(value))

    @field_validator("deployment_id")
    @classmethod
    def validate_deployment(cls, value: str) -> str:
        if value != PARAKEET_V3_CUDA.id:
            raise ValueError(f"Unsupported deployment_id: {value}")
        return value

    @field_validator("recording_prefix")
    @classmethod
    def validate_recording_prefix(cls, value: str) -> str:
        if not value or any(separator in value for separator in ("/", "\\")):
            raise ValueError("recording_prefix must be a nonempty filename prefix")
        return value

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, value: str) -> str:
        if value not in THEMES:
            raise ValueError(f"Unsupported Textual theme: {value}")
        return value


def config_path() -> Path:
    """Return UtilityHub Config's canonical VoicePad TOML path."""
    return get_config_path(APP_NAME, format="toml")


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate VoicePad settings through UtilityHub Config."""
    selected = path or config_path()
    if not selected.exists():
        return AppConfig()

    try:
        config, metadata = load_settings(
            AppConfig,
            app_name=APP_NAME,
            cwd=selected.parent,
            config_file=selected,
            env_vars=False,
        )
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        raise ConfigurationError(f"Could not load VoicePad configuration: {selected}") from error

    sources = sorted({source.source for source in metadata.per_field.values()})
    logger.info("Configuration loaded: path=%s sources=%s", selected, sources)
    return config


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    """Persist validated VoicePad settings through UtilityHub Config."""
    selected = path or config_path()
    try:
        saved = write_config(config, APP_NAME, path=selected, format="toml")
    except (OSError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Could not write VoicePad configuration: {selected}") from error
    logger.info("Configuration saved: path=%s", saved)
    return saved


__all__ = [
    "APP_NAME",
    "AppConfig",
    "ConfigurationError",
    "config_path",
    "load_config",
    "save_config",
]
