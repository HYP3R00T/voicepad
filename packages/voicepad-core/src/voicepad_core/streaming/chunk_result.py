# streaming/chunk_result.py

"""ChunkResult dataclass — the unit of output from StreamingTranscriber.

One ChunkResult is emitted per transcribed audio chunk via the on_chunk
callback. The final chunk always has is_final=True, even if it carries
no text (e.g. audio was too short to transcribe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..inference.types import Segment


@dataclass
class ChunkResult:
    """Result of transcribing one streaming audio chunk.

    Attributes:
        index:                Chunk sequence number, 1-based. Increments
                              with every dispatched chunk including empty ones.
        text:                 Transcribed text for this chunk. Empty string
                              when no speech was detected or audio was too short.
        segments:             Individual timed segments for this chunk.
                              Empty list when text is empty.
        start_s:              Logical start time of this chunk in seconds,
                              relative to the start of the full recording.
                              Does not include the overlap window.
        end_s:                End time of this chunk in seconds.
        latency_ms:           Wall-clock time taken to transcribe this chunk.
                              Zero for empty/error chunks.
        device:               Device used for this chunk ('cuda' or 'cpu').
        language:             BCP-47 language code detected by Whisper.
        language_probability: Confidence of the language detection (0.0–1.0).
        is_final:             True when this is the last chunk of the session.
                              Signals the caller that the stream is complete.
    """

    index: int
    text: str
    segments: list[Segment] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0
    latency_ms: float = 0.0
    device: str = "cuda"
    language: str = "en"
    language_probability: float = 1.0
    is_final: bool = False
