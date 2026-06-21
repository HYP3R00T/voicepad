from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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
