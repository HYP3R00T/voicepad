# inference/types.py

"""Data types for transcription results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WordTimestamp:
    """A single word with its timing and confidence.

    Attributes:
        word:        The word text (may include leading space).
        start:       Word start time in seconds.
        end:         Word end time in seconds.
        probability: Backend word confidence (0.0–1.0), when available.
    """

    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True, kw_only=True)
class Segment:
    """One transcription segment with timestamps and confidence metrics.

    Attributes:
        start:          Segment start time in seconds.
        end:            Segment end time in seconds.
        text:           Transcribed text for this segment.
        avg_logprob:    Segment log probability, when the backend reports it.
        no_speech_prob: No-speech probability, when the backend reports it.
        words:          Per-word timestamps. Empty list when word_timestamps=False.
        confidence:     Backend-neutral confidence (0.0–1.0), when available.
    """

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[WordTimestamp] = field(default_factory=list)
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Validate backend-neutral fields."""
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must be between 0.0 and 1.0"
            raise ValueError(msg)

    def duration(self) -> float:
        """Length of this segment in seconds."""
        return self.end - self.start


@dataclass(frozen=True, kw_only=True)
class TranscriptionResult:
    """Complete transcription result with metadata and quality metrics.

    Attributes:
        text:                  Full transcription text (all segments joined).
        segments:              Individual timed segments with confidence scores.
        language:              Detected language code, when available.
        language_probability:  Language confidence, when available.
        duration_s:            Duration of the audio that was transcribed.
        latency_ms:            Wall-clock time taken for the transcription call.
        device:                Device actually used ('cuda' or 'cpu').
        compute_type:          Precision used ('int8_float16', 'int8', etc.).
        fallback_to_cpu:       True if CUDA failed and CPU was used instead.
        backend_id:            Backend implementation that produced the result.
        model_id:              Model identifier reported by the backend.
        artifact_format:       Model artifact format, such as ``ct2`` or ``gguf``.

    ``device`` and ``compute_type`` describe runtime execution. The optional
    provenance fields identify which backend and model artifact produced the
    result without requiring model-family-specific metadata.
    """

    text: str
    segments: list[Segment]
    language: str | None
    language_probability: float | None
    duration_s: float
    latency_ms: float
    device: str
    compute_type: str
    fallback_to_cpu: bool = False
    backend_id: str | None = None
    model_id: str | None = None
    artifact_format: str | None = None
    audio_transformations: tuple[str, ...] = ()
    applied_options: tuple[str, ...] = ()
    ignored_options: tuple[str, ...] = ()
    language_source: str = "unavailable"
    word_timestamp_source: str = "unavailable"
    word_confidence_source: str = "unavailable"
    segment_log_probability_source: str = "unavailable"
    segment_confidence_source: str = "unavailable"
    no_speech_probability_source: str = "unavailable"
