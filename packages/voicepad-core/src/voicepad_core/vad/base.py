from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import SpeechSegment


class VADBase(ABC):
    """Abstract base class for all VAD implementations."""

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int = 16_000) -> list[SpeechSegment]: ...

    @abstractmethod
    def reset(self) -> None: ...
