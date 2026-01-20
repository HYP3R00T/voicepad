from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    recordings_path: Path = Path.home() / ".config" / "voicepad" / "recordings"
    markdown_path: Path = Path.home() / ".config" / "voicepad" / "markdown"
