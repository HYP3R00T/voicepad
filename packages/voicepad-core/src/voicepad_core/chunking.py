"""Real-time audio chunking using faster-whisper's Silero VAD.

This module provides smart audio chunking for long recordings by detecting
natural speech boundaries using Voice Activity Detection (VAD). Chunks are
created when:
1. Minimum duration threshold is reached (e.g., 60 seconds)
2. A natural silence boundary is detected

This enables background transcription while recording continues, reducing
overall latency for long recordings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for an audio chunk.

    Attributes:
        index: Sequential chunk index (0-based).
        start_time: Start time in seconds from recording start.
        duration: Duration of the chunk in seconds.
        sample_count: Number of audio samples in the chunk.
    """

    index: int
    start_time: float
    duration: float
    sample_count: int


class RealtimeChunker:
    """Manages real-time audio chunking with VAD-based segmentation.

    This class implements a rolling buffer strategy:
    1. Continuously accumulates incoming audio frames
    2. Once minimum duration is reached, analyzes buffer with Silero VAD
    3. Identifies natural speech boundaries (speech followed by silence)
    4. Extracts complete chunk and returns it for transcription
    5. Retains remaining audio in buffer for next chunk

    The approach balances real-time processing with context preservation.
    VAD analysis runs periodically (not per-frame) to minimize overhead.

    Example:
        >>> chunker = RealtimeChunker(min_chunk_duration=60.0)
        >>> # In audio callback loop:
        >>> chunk = chunker.add_audio(audio_data)
        >>> if chunk is not None:
        >>>     transcribe_chunk(chunk)
    """

    def __init__(
        self,
        min_chunk_duration: float = 60.0,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 1000,
        speech_pad_ms: int = 400,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the real-time chunker.

        Args:
            min_chunk_duration: Minimum duration (seconds) before allowing chunk splits.
                Ensures sufficient context for accurate transcription.
            vad_threshold: Silero VAD speech probability threshold (0.0-1.0).
                Higher values = more strict (less likely to detect as speech).
            min_silence_duration_ms: Minimum silence duration (ms) to trigger chunk boundary.
                Ensures clean breaks between chunks.
            speech_pad_ms: Padding (ms) added to each side of detected speech segments.
                Prevents cutting off speech edges.
            sample_rate: Audio sample rate in Hz.
        """
        # Validate parameters
        if min_chunk_duration < 10.0:
            msg = f"min_chunk_duration must be >= 10 seconds, got {min_chunk_duration}"
            raise ValueError(msg)
        if not 0.0 <= vad_threshold <= 1.0:
            msg = f"vad_threshold must be between 0.0 and 1.0, got {vad_threshold}"
            raise ValueError(msg)
        if min_silence_duration_ms < 100:
            msg = f"min_silence_duration_ms must be >= 100ms, got {min_silence_duration_ms}"
            raise ValueError(msg)

        self.min_chunk_duration = min_chunk_duration
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        # Rolling buffer state
        self._buffer: list[np.ndarray] = []
        self._buffer_duration: float = 0.0  # Duration in seconds
        self._total_processed_duration: float = 0.0  # Total audio processed
        self._chunk_index: int = 0

        # Lazy load VAD only when needed
        self._vad_model = None

        logger.info(
            "RealtimeChunker initialized: min_duration=%.1fs, vad_threshold=%.2f, min_silence=%dms, sample_rate=%d",
            min_chunk_duration,
            vad_threshold,
            min_silence_duration_ms,
            sample_rate,
        )

    def _get_vad_model(self):
        """Lazy load Silero VAD model from faster-whisper."""
        if self._vad_model is None:
            from faster_whisper.vad import get_vad_model

            self._vad_model = get_vad_model()
            logger.debug("Loaded Silero VAD model")

        return self._vad_model

    def _concatenate_buffer(self) -> np.ndarray:
        """Concatenate all audio frames in buffer into single array.

        Returns 2D array to maintain dimensional consistency in buffer.
        Flatten only when needed (for VAD or saving).
        """
        if not self._buffer:
            return np.array([], dtype=np.float32)

        # Keep as 2D for consistent buffer management
        # Sounddevice provides (frames, channels), we preserve this shape
        return np.concatenate(self._buffer)

    def _analyze_buffer_for_boundary(self) -> int | None:
        """Analyze buffer with VAD to find a suitable chunk boundary.

        Returns:
            Sample index where chunk should end, or None if no boundary found.
            The boundary is at the end of the last speech segment that has
            sufficient silence after it.
        """
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        # Get full buffer as numpy array
        buffer_audio = self._concatenate_buffer()

        if len(buffer_audio) == 0:
            return None

        # Configure VAD options
        vad_options = VadOptions(
            threshold=self.vad_threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

        # Get speech timestamps from Silero VAD
        # VAD expects 1D array, so flatten the 2D buffer
        try:
            speech_timestamps = get_speech_timestamps(
                buffer_audio.flatten(),  # Flatten (n, 1) -> (n) for VAD
                vad_options=vad_options,
                sampling_rate=self.sample_rate,
            )
        except Exception as e:
            logger.warning("VAD analysis failed: %s", e)
            return None

        if not speech_timestamps:
            # No speech detected in buffer - unusual but possible
            logger.debug("No speech detected in buffer")
            return None

        # Find the last speech segment that ends with sufficient silence before buffer end
        buffer_samples = len(buffer_audio)

        for speech_segment in reversed(speech_timestamps):
            speech_end = speech_segment["end"]
            silence_after = buffer_samples - speech_end

            # Check if there's enough silence after this speech segment
            silence_duration_ms = (silence_after / self.sample_rate) * 1000

            if silence_duration_ms >= self.min_silence_duration_ms:
                # Found a good boundary
                logger.debug(
                    "Found chunk boundary at sample %d (%.2fs into buffer)",
                    speech_end,
                    speech_end / self.sample_rate,
                )
                return speech_end

        # No suitable boundary found yet
        logger.debug("No suitable chunk boundary found yet")
        return None

    def add_audio(self, audio_data: np.ndarray) -> tuple[np.ndarray, ChunkMetadata] | None:
        """Add audio data to the chunker buffer.

        This method should be called for each incoming audio frame. It accumulates
        audio in a rolling buffer and periodically checks for natural chunk boundaries
        using VAD.

        Args:
            audio_data: Audio data as numpy array (float32, mono).

        Returns:
            Tuple of (chunk_audio, metadata) if a chunk is ready, None otherwise.
        """
        # Add to buffer
        self._buffer.append(audio_data.copy())
        frame_duration = len(audio_data) / self.sample_rate
        self._buffer_duration += frame_duration

        # Check if we've reached minimum chunk duration
        if self._buffer_duration < self.min_chunk_duration:
            return None

        # Analyze buffer for a natural boundary
        boundary_sample = self._analyze_buffer_for_boundary()

        if boundary_sample is None:
            # No boundary found yet, keep accumulating
            return None

        # Extract chunk up to the boundary
        buffer_audio = self._concatenate_buffer()
        chunk_audio = buffer_audio[:boundary_sample]
        remaining_audio = buffer_audio[boundary_sample:]

        # Create metadata
        chunk_duration = len(chunk_audio) / self.sample_rate
        metadata = ChunkMetadata(
            index=self._chunk_index,
            start_time=self._total_processed_duration,
            duration=chunk_duration,
            sample_count=len(chunk_audio),
        )

        logger.info(
            "Chunk %d ready: %.2fs duration, %.2f-%.2fs in recording",
            metadata.index,
            chunk_duration,
            metadata.start_time,
            metadata.start_time + chunk_duration,
        )

        # Update state for next chunk
        self._chunk_index += 1
        self._total_processed_duration += chunk_duration

        # Reset buffer with remaining audio (keep as 2D)
        self._buffer = [remaining_audio] if len(remaining_audio) > 0 else []
        self._buffer_duration = len(remaining_audio) / self.sample_rate

        # Return flattened chunk for saving (soundfile needs 1D)
        return chunk_audio.flatten(), metadata

    def finalize(self) -> tuple[np.ndarray, ChunkMetadata] | None:
        """Get any remaining audio in the buffer as the final chunk.

        This should be called when recording ends to ensure all audio is processed.

        Returns:
            Tuple of (chunk_audio, metadata) if buffer has audio, None otherwise.
        """
        if not self._buffer or self._buffer_duration == 0:
            logger.debug("No remaining audio to finalize")
            return None

        # Get all remaining audio
        chunk_audio = self._concatenate_buffer()
        chunk_duration = len(chunk_audio) / self.sample_rate

        metadata = ChunkMetadata(
            index=self._chunk_index,
            start_time=self._total_processed_duration,
            duration=chunk_duration,
            sample_count=len(chunk_audio),
        )

        logger.info(
            "Final chunk %d: %.2fs duration, %.2f-%.2fs in recording",
            metadata.index,
            chunk_duration,
            metadata.start_time,
            metadata.start_time + chunk_duration,
        )

        # Clear buffer
        self._buffer = []
        self._buffer_duration = 0.0

        # Return flattened chunk for saving (soundfile needs 1D)
        return chunk_audio.flatten(), metadata

    def reset(self) -> None:
        """Reset the chunker state.

        Clears the buffer and resets all counters. Useful for starting a new
        recording session with the same chunker instance.
        """
        self._buffer = []
        self._buffer_duration = 0.0
        self._total_processed_duration = 0.0
        self._chunk_index = 0

        logger.debug("Chunker state reset")
