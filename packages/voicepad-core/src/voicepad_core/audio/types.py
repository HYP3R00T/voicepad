from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFormat:
    """Audio stream metadata shared across raw audio sources."""

    sample_rate: int
    channels: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")


@dataclass(frozen=True)
class RawAudio:
    """Captured or loaded audio with its source metadata."""

    samples: np.ndarray
    sample_rate: int
    channels: int

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")

    def duration(self) -> float:
        """Length of the audio in seconds."""
        return 0.0 if self.sample_rate == 0 else float(self.samples.shape[0]) / self.sample_rate

    def audio_format(self) -> AudioFormat:
        """Return the raw audio format metadata."""
        return AudioFormat(sample_rate=self.sample_rate, channels=self.channels)


__all__ = ["AudioFormat", "RawAudio"]
