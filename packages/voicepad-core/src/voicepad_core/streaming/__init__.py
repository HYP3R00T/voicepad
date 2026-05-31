# streaming/__init__.py

"""Streaming transcription package.

Provides VAD-triggered chunked transcription over a live AudioRecorder
buffer. Chunks are delivered via callback as they are transcribed, so
the caller sees partial results during recording rather than waiting
for the full recording to complete.

Quick start:
    from streaming import StreamingTranscriber, ChunkResult

    streamer = StreamingTranscriber(
        recorder=recorder,          # started AudioRecorder instance
        on_chunk=handle_chunk,      # Callable[[ChunkResult], None]
        on_error=handle_error,      # Callable[[str], None]
    )
    streamer.start()
    # ... recording happens ...
    streamer.stop()                 # blocks until final chunk is delivered
"""

from .chunk_result import ChunkResult
from .transcriber import MIN_CHUNK_S, OVERLAP_S, POLL_INTERVAL_S, StreamingTranscriber

__all__ = [
    "ChunkResult",
    "StreamingTranscriber",
    # Constants — exposed so callers can read defaults without importing internals
    "MIN_CHUNK_S",
    "OVERLAP_S",
    "POLL_INTERVAL_S",
]
