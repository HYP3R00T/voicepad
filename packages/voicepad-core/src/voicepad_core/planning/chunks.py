from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from voicepad_core.vad import NaturalPause

SAMPLE_RATE = 16_000


class OverlapKind(StrEnum):
    NONE = "none"
    NATURAL = "natural"
    CAPPED_NATURAL = "capped-natural"
    FORCED = "forced"


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    minimum_boundary_seconds: int = 20
    natural_search_seconds: int = 25
    preferred_seconds: int = 30
    maximum_semantic_overlap_seconds: int = 12
    forced_overlap_seconds: int = 2
    maximum_input_seconds: int = 60

    def __post_init__(self) -> None:
        if not 0 < self.minimum_boundary_seconds <= self.natural_search_seconds <= self.preferred_seconds:
            raise ValueError("chunk boundary timings are inconsistent")
        if not 0 < self.forced_overlap_seconds <= self.maximum_semantic_overlap_seconds:
            raise ValueError("chunk overlap timings are inconsistent")
        if self.preferred_seconds + self.maximum_semantic_overlap_seconds > self.maximum_input_seconds:
            raise ValueError("preferred chunk and semantic overlap exceed the hard input limit")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    source_start_sample: int
    source_end_sample: int
    logical_start_sample: int
    logical_end_sample: int
    overlap: OverlapKind
    terminal: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.source_start_sample <= self.logical_start_sample < self.logical_end_sample:
            raise ValueError("chunk source/logical start positions are invalid")
        if self.logical_end_sample != self.source_end_sample:
            raise ValueError("chunk logical and source ends must match")


class AdaptiveChunkPlanner:
    """Plan bounded descriptors from confirmed pause history and committed audio."""

    def __init__(self, policy: ChunkPolicy | None = None) -> None:
        self.policy = policy or ChunkPolicy()
        self._pauses: list[NaturalPause] = []
        self._logical_start = 0
        self._source_start = 0
        self._overlap = OverlapKind.NONE

    def add_pause(self, pause: NaturalPause) -> None:
        breakpoint = pause.breakpoint_sample
        if self._pauses and breakpoint <= self._pauses[-1].breakpoint_sample:
            raise ValueError("natural pauses must be strictly ordered")
        self._pauses.append(pause)

    def poll(self, committed_samples: int, *, final: bool = False) -> tuple[AudioChunk, ...]:
        if committed_samples < self._logical_start:
            raise ValueError("committed sample cursor moved backwards")
        chunks: list[AudioChunk] = []
        while committed_samples > self._logical_start:
            hard_end = self._source_start + self.policy.maximum_input_seconds * SAMPLE_RATE
            search_start = self._logical_start + self.policy.natural_search_seconds * SAMPLE_RATE
            boundary = None
            if committed_samples >= search_start:
                boundary = self._natural_boundary(search_start, min(committed_samples, hard_end))
            if boundary is not None:
                chunks.append(self._emit(boundary, natural=True))
                continue
            if committed_samples >= hard_end:
                chunks.append(self._emit(hard_end, natural=False))
                continue
            break

        if final and committed_samples > self._logical_start:
            chunks.append(
                AudioChunk(
                    self._source_start,
                    committed_samples,
                    self._logical_start,
                    committed_samples,
                    self._overlap,
                    terminal=True,
                )
            )
            self._logical_start = committed_samples
            self._source_start = committed_samples
            self._overlap = OverlapKind.NONE
        return tuple(chunks)

    def _natural_boundary(self, search_start: int, committed: int) -> int | None:
        earliest = max(
            self._logical_start + self.policy.minimum_boundary_seconds * SAMPLE_RATE,
            search_start,
        )
        candidates = [
            pause.breakpoint_sample for pause in self._pauses if earliest <= pause.breakpoint_sample <= committed
        ]
        return candidates[0] if candidates else None

    def _emit(self, boundary: int, *, natural: bool) -> AudioChunk:
        chunk = AudioChunk(
            self._source_start,
            boundary,
            self._logical_start,
            boundary,
            self._overlap,
        )
        previous_logical_start = self._logical_start
        self._logical_start = boundary
        if natural:
            prior_breakpoints = [
                pause.breakpoint_sample
                for pause in self._pauses
                if previous_logical_start <= pause.breakpoint_sample < boundary
            ]
            cap = self.policy.maximum_semantic_overlap_seconds * SAMPLE_RATE
            if prior_breakpoints and boundary - prior_breakpoints[-1] <= cap:
                self._source_start = prior_breakpoints[-1]
                self._overlap = OverlapKind.NATURAL
            else:
                self._source_start = max(previous_logical_start, boundary - cap)
                self._overlap = OverlapKind.CAPPED_NATURAL
        else:
            overlap = self.policy.forced_overlap_seconds * SAMPLE_RATE
            self._source_start = max(previous_logical_start, boundary - overlap)
            self._overlap = OverlapKind.FORCED
        return chunk
