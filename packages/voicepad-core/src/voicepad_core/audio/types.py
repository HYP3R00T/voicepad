from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WaveformSpec:
    """Canonical waveform shape required by a processing deployment."""

    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")


@dataclass(frozen=True)
class AudioWindow:
    """A contiguous waveform range addressed in source sample positions."""

    samples: np.ndarray
    start_sample: int

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if self.samples.ndim != 1:
            raise ValueError("audio window must be mono")
        if self.start_sample < 0:
            raise ValueError("start_sample must not be negative")
        self.samples.setflags(write=False)

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.samples)


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
        if self.samples.ndim not in (1, 2):
            raise ValueError("samples must be a mono vector or frame-by-channel matrix")
        if self.samples.ndim == 1 and self.channels != 1:
            raise ValueError("multi-channel samples must use a frame-by-channel matrix")
        if self.samples.ndim == 2 and self.samples.shape[1] != self.channels:
            raise ValueError("sample shape does not match channel metadata")
        self.samples.setflags(write=False)

    def duration(self) -> float:
        """Length of the audio in seconds."""
        return self.frame_count / self.sample_rate

    @property
    def frame_count(self) -> int:
        """Return the number of sample frames, independent of storage layout."""
        return int(self.samples.shape[0])


__all__ = ["AudioWindow", "RawAudio", "WaveformSpec"]
