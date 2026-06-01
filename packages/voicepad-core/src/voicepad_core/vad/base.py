# vad/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechSegment:
    """
    A single speech region detected by a VAD pass.

    start and end are in seconds, relative to the
    beginning of the audio array passed to detect().
    """

    start: float  # seconds
    end: float  # seconds

    def duration(self) -> float:
        """Length of this speech segment in seconds."""
        return self.end - self.start

    def to_sample_indices(self, sample_rate: int) -> tuple[int, int]:
        """
        Convert start/end times to integer sample indices.

        Args:
            sample_rate: Sample rate of the audio (typically 16000).

        Returns:
            (start_sample, end_sample) as integers.
        """
        return (
            int(self.start * sample_rate),
            int(self.end * sample_rate),
        )


class VADBase(ABC):
    """
    Abstract base class for all VAD implementations.

    Accept a 16kHz mono float32 numpy array.
    Return a list of SpeechSegment objects.

    StreamingTranscriber depends only on VADBase —
    never on a concrete implementation.
    """

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int = 16_000) -> list[SpeechSegment]: ...

    @abstractmethod
    def reset(self) -> None: ...
