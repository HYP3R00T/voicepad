from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechSegment:
    """A single speech region detected by a VAD pass."""

    start: float
    end: float

    def duration(self) -> float:
        return self.end - self.start

    def to_sample_indices(self, sample_rate: int) -> tuple[int, int]:
        return (int(self.start * sample_rate), int(self.end * sample_rate))


__all__ = ["SpeechSegment"]
