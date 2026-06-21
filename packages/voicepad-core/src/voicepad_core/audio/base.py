from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import RawAudio


class AudioSource(ABC):
    """Abstract base class for raw audio sources."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return raw float32 audio samples from the source."""

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return the source's native sample rate in Hz."""

    @abstractmethod
    def get_channels(self) -> int:
        """Return the number of source channels."""

    def read_audio(self) -> RawAudio:
        """Return raw audio bundled with its source metadata."""
        return RawAudio(
            samples=self.read(),
            sample_rate=self.get_sample_rate(),
            channels=self.get_channels(),
        )
