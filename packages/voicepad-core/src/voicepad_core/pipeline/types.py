from __future__ import annotations

from dataclasses import dataclass

from voicepad_core.inference import ActiveDeployment
from voicepad_core.planning import AudioChunk
from voicepad_core.vad import SpeechRegion


@dataclass(frozen=True, slots=True)
class ObservedToken:
    text: str
    start_sample: int
    end_sample: int
    chunk_index: int


@dataclass(frozen=True, slots=True)
class ObservedWord:
    text: str
    start_sample: int
    end_sample: int
    chunk_index: int
    physical_start_sample: int
    physical_end_sample: int

    @property
    def midpoint_sample(self) -> int:
        return (self.start_sample + self.end_sample) // 2

    @property
    def edge_distance_samples(self) -> int:
        return min(
            self.midpoint_sample - self.physical_start_sample,
            self.physical_end_sample - self.midpoint_sample,
        )


@dataclass(frozen=True, slots=True)
class CoverageGap:
    speech: SpeechRegion
    reason: str


@dataclass(frozen=True, slots=True)
class ChunkOutcome:
    index: int
    descriptor: AudioChunk
    latency_seconds: float
    token_count: int
    word_count: int
    cancelled: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FileTranscriptionResult:
    text: str
    words: tuple[ObservedWord, ...]
    tokens: tuple[ObservedToken, ...]
    duration_seconds: float
    latency_seconds: float
    deployment: ActiveDeployment
    chunks: tuple[ChunkOutcome, ...]
    speech_regions: tuple[SpeechRegion, ...]
    coverage_gaps: tuple[CoverageGap, ...]
    warnings: tuple[str, ...]
    complete: bool
