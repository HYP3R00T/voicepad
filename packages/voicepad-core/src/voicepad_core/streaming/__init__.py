# streaming/__init__.py

"""Streaming transcription package.

Provides VAD-triggered chunked transcription over a live MicrophoneStream
buffer. Chunks are delivered via callback as they are transcribed, so
the caller sees partial results during recording rather than waiting
for the full recording to complete.

Quick start:
    from voicepad_core.streaming import StreamingTranscriber, ChunkResult

    streamer = StreamingTranscriber(
        recorder=mic_stream,        # started MicrophoneStream instance
        on_chunk=handle_chunk,      # Callable[[ChunkResult], None]
        on_error=handle_error,      # Callable[[str], None]
    )
    streamer.start()
    # ... recording happens ...
    streamer.stop()                 # blocks until final chunk is delivered
"""

from .chunk_result import ChunkResult
from .transcriber import (
    MAX_CHUNK_S,
    MIN_CHUNK_S,
    OVERLAP_S,
    POLL_INTERVAL_S,
    SILENCE_THRESHOLD_MS,
    StreamingTranscriber,
)

__all__ = [
    "ChunkResult",
    "StreamingTranscriber",
    # Constants — exposed so callers can read defaults without importing internals
    "MIN_CHUNK_S",
    "MAX_CHUNK_S",
    "OVERLAP_S",
    "POLL_INTERVAL_S",
    "SILENCE_THRESHOLD_MS",
]
