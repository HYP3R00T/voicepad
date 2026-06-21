"""Core configuration loading helpers for voicepad."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utilityhub_config import load_settings

from .types import Config


def get_config(cwd: Path | None = None, app_name: str = "voicepad") -> Config:
    settings, _ = load_settings(Config, app_name=app_name, cwd=cwd)
    return settings


def get_config_with_metadata(cwd: Path | None = None, app_name: str = "voicepad") -> tuple[Config, Any]:
    return load_settings(Config, app_name=app_name, cwd=cwd)


__all__ = ["get_config", "get_config_with_metadata", "Config"]
