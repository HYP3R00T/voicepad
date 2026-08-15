"""Behavioral contracts shared by the pipeline coordinators."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from voicepad_core.inference import (
    ActiveDeployment,
    BackendResult,
    CancellationToken,
    TranscriptionIntent,
)
from voicepad_core.preprocessing import PreprocessedAudio
from voicepad_core.vad import VadFrame


class ReadyEngine(Protocol):
    """Engine behavior required to transcribe one prepared audio chunk."""

    @property
    def active_deployment(self) -> ActiveDeployment | None:
        """Return the active deployment, or ``None`` when the engine is not ready."""
        ...

    def transcribe(
        self,
        audio: PreprocessedAudio,
        intent: TranscriptionIntent | None = None,
        cancellation: CancellationToken | None = None,
    ) -> BackendResult:
        """Transcribe one canonical, bounded audio input."""
        ...


class SequentialVad(Protocol):
    """Voice-activity detector that consumes audio in sequential sample order."""

    def reset(self) -> None:
        """Clear context retained from a previous audio stream."""
        ...

    def accept(
        self,
        samples: np.ndarray,
        start_sample: int,
        *,
        final: bool = False,
    ) -> tuple[VadFrame, ...]:
        """Analyze the next contiguous range and return absolute VAD frames."""
        ...


__all__ = ["ReadyEngine", "SequentialVad"]
