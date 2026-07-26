"""Backend-neutral contracts for model preparation and transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..audio.types import WaveformSpec

if TYPE_CHECKING:
    from .types import TranscriptionResult
    from ..models import ModelSpec

DecodingIntent = Literal["transcribe", "translate"]
OutputProvenance = Literal["unavailable", "native", "derived"]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Features implemented by a backend driver."""

    streaming: bool = False
    word_timestamps: bool = False
    language_detection: bool = False
    translation: bool = False
    context_biasing: bool = False
    language_selection: bool = False
    beam_search: bool = False
    text_prompt: bool = False
    vad_filter: bool = False
    no_speech_threshold: bool = False
    hallucination_silence_threshold: bool = False


@dataclass(frozen=True, slots=True)
class OutputCapabilities:
    """Provenance of optional values emitted by a backend."""

    language: OutputProvenance = "unavailable"
    word_timestamps: OutputProvenance = "unavailable"
    word_confidence: OutputProvenance = "unavailable"
    segment_log_probability: OutputProvenance = "unavailable"
    segment_confidence: OutputProvenance = "unavailable"
    no_speech_probability: OutputProvenance = "unavailable"


@dataclass(frozen=True, slots=True)
class BackendContract:
    """Complete handover contract published by one backend driver."""

    audio: WaveformSpec
    decoding: BackendCapabilities = field(default_factory=BackendCapabilities)
    output: OutputCapabilities = field(default_factory=OutputCapabilities)


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Requested execution settings used when opening a model."""

    device: str = "auto"
    precision: str = "auto"
    allow_cpu_fallback: bool = True

    def __post_init__(self) -> None:
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if not self.precision.strip():
            raise ValueError("precision must not be empty")


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Actual execution settings reported by an open session."""

    backend_id: str
    model_id: str
    device: str
    precision: str
    fallback_to_cpu: bool = False

    def __post_init__(self) -> None:
        values = {
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "device": self.device,
            "precision": self.precision,
        }
        for name, value in values.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class TranscriptionContext:
    """Semantic hints supplied to a backend without rewriting its output."""

    proper_nouns: tuple[str, ...] = ()
    previous_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "proper_nouns", tuple(self.proper_nouns))
        if any(not term.strip() for term in self.proper_nouns):
            raise ValueError("proper_nouns must not contain empty terms")


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    """Canonical audio and decoding intent passed to an open session."""

    audio: NDArray[np.float32]
    sample_rate: int
    language: str | None = None
    word_timestamps: bool = False
    beam_size: int | None = 5
    intent: DecodingIntent = "transcribe"
    context: TranscriptionContext = field(default_factory=TranscriptionContext)
    vad_filter: bool | None = False
    no_speech_threshold: float | None = 0.6
    hallucination_silence_threshold: float | None = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.audio, np.ndarray):
            raise TypeError("audio must be a numpy.ndarray")
        if self.audio.ndim != 1:
            raise ValueError("audio must be mono")
        if self.audio.dtype != np.float32:
            raise ValueError("audio must use float32 samples")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be empty")
        if self.beam_size is not None and self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")
        if self.intent not in ("transcribe", "translate"):
            raise ValueError(f"unsupported decoding intent: {self.intent}")
        if self.no_speech_threshold is not None and not 0.0 <= self.no_speech_threshold <= 1.0:
            raise ValueError("no_speech_threshold must be between 0.0 and 1.0")
        if self.hallucination_silence_threshold is not None and self.hallucination_silence_threshold < 0.0:
            raise ValueError("hallucination_silence_threshold must not be negative")


@dataclass(frozen=True, slots=True)
class PreparedModel:
    """A model artifact prepared for a backend to open."""

    spec: ModelSpec
    artifact_path: Path


@runtime_checkable
class TranscriptionSession(Protocol):
    """An open model session that performs repeated inference calls."""

    @property
    def info(self) -> RuntimeInfo: ...

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...

    def close(self) -> None: ...


@runtime_checkable
class BackendDriver(Protocol):
    """A backend capable of preparing and opening model sessions."""

    @property
    def id(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def contract(self) -> BackendContract: ...

    def is_available(self) -> bool: ...

    def prepare(self, model: ModelSpec) -> PreparedModel: ...

    def open(self, model: PreparedModel, options: RuntimeOptions) -> TranscriptionSession: ...


__all__ = [
    "BackendCapabilities",
    "BackendContract",
    "BackendDriver",
    "DecodingIntent",
    "OutputCapabilities",
    "OutputProvenance",
    "PreparedModel",
    "RuntimeInfo",
    "RuntimeOptions",
    "TranscriptionContext",
    "TranscriptionRequest",
    "TranscriptionSession",
]
