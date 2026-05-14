"""Data types for transcription results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """Transcription segment with timestamps and confidence metrics.

    Attributes:
        start: Segment start time in seconds
        end: Segment end time in seconds
        text: Transcribed text for this segment
        avg_logprob: Average log probability (confidence score, typically -inf to 0)
        no_speech_prob: Probability that segment contains no speech (0.0 to 1.0)
    """

    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete transcription result with metadata and quality metrics.

    Attributes:
        text: Full transcription text (all segments joined)
        segments: Individual timed segments with confidence scores
        language: Language code (e.g., "en", "es")
        language_probability: Language detection confidence (0.0 to 1.0)
        duration_s: Audio duration in seconds
        latency_ms: Processing time in milliseconds
        device: Device used for inference ("cuda" or "cpu")
        compute_type: Precision used (e.g., "int8", "float16")
        fallback_to_cpu: Whether GPU inference failed and fell back to CPU
        avg_confidence: Mean confidence across all segments
        low_confidence_segments: Count of segments with confidence below -1.0
    """

    text: str
    segments: list[Segment]
    language: str
    language_probability: float
    duration_s: float
    latency_ms: float
    device: str
    compute_type: str
    fallback_to_cpu: bool = False
    avg_confidence: float = 0.0
    low_confidence_segments: int = 0
