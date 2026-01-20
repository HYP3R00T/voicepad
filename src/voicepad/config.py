from pathlib import Path

from pydantic import BaseModel
from utilityhub_config import load_settings


class Config(BaseModel):
    recordings_path: Path = Path.home() / ".config" / "voicepad" / "recordings"
    markdown_path: Path = Path.home() / ".config" / "voicepad" / "markdown"


def get_config() -> Config:
    config, _ = load_settings(Config, app_name="voicepad")
    return config
