"""Configuration management for Voicepad."""

from pathlib import Path

from pydantic import BaseModel
from utilityhub_config import load_settings


class Config(BaseModel):
    """Configuration for Voicepad."""

    recordings_path: Path = Path("data/recordings")
    markdown_path: Path = Path("data/markdown")


def get_config() -> Config:
    """Load and return configuration from voicepad.yaml, environment, or defaults."""
    config, _ = load_settings(Config, app_name="voicepad")
    return config
