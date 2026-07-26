from __future__ import annotations

from dataclasses import dataclass, field

from ..inference.types import Segment


@dataclass(frozen=True)
class ChunkResult:
    """Result of transcribing one streaming audio chunk."""

    index: int
    text: str
    segments: list[Segment] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0
    latency_ms: float = 0.0
    device: str = "cuda"
    language: str | None = None
    language_probability: float | None = None
    is_final: bool = False


__all__ = ["ChunkResult"]
