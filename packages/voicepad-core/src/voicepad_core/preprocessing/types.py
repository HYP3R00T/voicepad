from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MONO_CHANNELS, TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class PreprocessedAudio:
    """Whisper-ready mono float32 audio."""

    samples: np.ndarray
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = MONO_CHANNELS

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels != MONO_CHANNELS:
            raise ValueError(f"preprocessed audio must be mono, got {self.channels} channels")

    def duration(self) -> float:
        """Length of the audio in seconds."""
        return 0.0 if self.sample_rate == 0 else float(self.samples.shape[0]) / self.sample_rate


__all__ = ["PreprocessedAudio"]
