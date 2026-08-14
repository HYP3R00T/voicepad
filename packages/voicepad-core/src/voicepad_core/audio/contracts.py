from __future__ import annotations

from typing import Protocol

from .types import AudioWindow


class IncrementalAudioSource(Protocol):
    """Committed audio ranges that may continue growing until finalized."""

    @property
    def sample_rate(self) -> int: ...

    @property
    def channels(self) -> int: ...

    @property
    def committed_samples(self) -> int: ...

    @property
    def is_final(self) -> bool: ...

    def read_range(self, start_sample: int, end_sample: int) -> AudioWindow: ...

    def wait_for_update(self, after_sample: int, timeout: float | None = None) -> tuple[int, bool]: ...


__all__ = ["IncrementalAudioSource"]
