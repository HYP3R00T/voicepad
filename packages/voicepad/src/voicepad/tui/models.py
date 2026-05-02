"""Data models for the VoicePad TUI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SessionEntry:
    index: int
    wav_path: Path | None
    md_path: Path | None
    duration_s: float
    text: str
    latency_ms: float
    device: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
