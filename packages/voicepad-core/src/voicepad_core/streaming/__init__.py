"""Streaming transcription package."""

from .constants import MAX_CHUNK_S, MIN_CHUNK_S, OVERLAP_S, POLL_INTERVAL_S, SILENCE_THRESHOLD_MS
from .errors import StreamingConfigurationError, StreamingError, StreamingRecorderError
from .transcriber import StreamingTranscriber
from .types import ChunkResult

__all__ = [
    "ChunkResult",
    "StreamingTranscriber",
    "StreamingError",
    "StreamingConfigurationError",
    "StreamingRecorderError",
    "MIN_CHUNK_S",
    "MAX_CHUNK_S",
    "OVERLAP_S",
    "POLL_INTERVAL_S",
    "SILENCE_THRESHOLD_MS",
]
