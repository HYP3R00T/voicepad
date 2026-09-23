"""Non-destructive signal measurements, computed outside the capture callback."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalHealth:
    """Immutable snapshot of captured samples, before PCM16 conversion.

    Level warnings describe samples, not microphone availability or speech detection.
    RMS excludes non-finite samples, which are reported separately.
    """

    frames: int = 0
    samples: int = 0
    duration_s: float = 0.0
    peak: float = 0.0
    sum_squares: float = 0.0
    latest_rms: float = 0.0
    pcm_limit_samples: int = 0
    nonfinite_samples: int = 0

    @property
    def rms(self) -> float:
        finite_count = self.samples - self.nonfinite_samples
        return math.sqrt(self.sum_squares / finite_count) if finite_count else 0.0

    @property
    def peak_dbfs(self) -> float:
        return 20 * math.log10(self.peak) if self.peak else -math.inf

    @property
    def rms_dbfs(self) -> float:
        return 20 * math.log10(self.rms) if self.rms else -math.inf

    @property
    def latest_rms_dbfs(self) -> float:
        return 20 * math.log10(self.latest_rms) if self.latest_rms else -math.inf

    @property
    def warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if self.nonfinite_samples:
            warnings.append("Audio contains non-finite samples; the recording may be invalid.")
        if self.pcm_limit_samples:
            warnings.append("Audio reached PCM16 limits; possible clipping. Check microphone gain.")
        if self.duration_s >= 2.0 and not self.nonfinite_samples and self.rms < 0.001:
            warnings.append(
                "Audio is mostly near-silent (RMS below -60 dBFS). Check input, mute, and gain if unexpected."
            )
        return tuple(warnings)

    def with_samples(self, samples: np.ndarray, sample_rate: int) -> SignalHealth:
        """Accumulate a writer block without modifying it or retaining sample data."""
        if samples.size == 0:
            return self
        values = np.asarray(samples, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        sum_squares = float(np.dot(finite, finite))
        frames = self.frames + len(samples)
        return SignalHealth(
            frames=frames,
            samples=self.samples + samples.size,
            duration_s=frames / sample_rate,
            peak=max(self.peak, float(np.max(np.abs(finite))) if finite.size else 0.0),
            sum_squares=self.sum_squares + sum_squares,
            latest_rms=math.sqrt(sum_squares / finite.size) if finite.size else 0.0,
            pcm_limit_samples=self.pcm_limit_samples
            + int(np.count_nonzero((finite >= 32767 / 32768) | (finite <= -1.0))),
            nonfinite_samples=self.nonfinite_samples + values.size - finite.size,
        )
