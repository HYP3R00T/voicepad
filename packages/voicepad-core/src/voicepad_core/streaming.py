"""VAD-triggered streaming transcription.

Runs alongside an active AudioRecorder. Monitors the audio buffer in real time,
detects silence boundaries using a lightweight energy-based VAD, and dispatches
chunks to the Whisper model as they accumulate — so transcription happens
*during* recording rather than after.

Architecture:
    AudioRecorder._frames  ←  sounddevice callback (16 kHz, continuous)
           ↓  (polled every 0.3s)
    StreamingTranscriber   ←  background thread
      - accumulates audio
      - when silence detected after MIN_CHUNK_S of speech → dispatch chunk
      - transcribes chunk → on_chunk callback → TUI updates live
      - on stop() → transcribes remaining tail → on_chunk(is_final=True)

Result: for a 5-minute recording, user waits ~1.3s after stopping
        instead of ~15s for the full batch.

Note: MIN_CHUNK_S is aligned with retranscription's vad_min_chunk_duration (10s)
      to ensure consistent chunking behavior between streaming and retranscription modes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from voicepad_core.audio import AudioRecorder
    from voicepad_core.config import Config
    from voicepad_core.transcription import Segment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Minimum pending audio before we consider splitting (seconds)
# 30s ensures chunks contain complete thoughts; avoids mid-sentence splits.
MIN_CHUNK_S: float = 30.0

# Silence detection: RMS below this = silence
SILENCE_RMS_THRESHOLD: float = 0.01

# Silence must persist this long to trigger a split (seconds)
# Aligned with retranscription's vad_min_silence_duration_ms (1000ms = 1.0s)
SILENCE_TRIGGER_S: float = 1.0

# Poll interval for the monitor thread (seconds)
POLL_INTERVAL_S: float = 0.3

# Audio overlap included from previous chunk for acoustic context (seconds)
# Kept short — just enough for the model to not lose context at the boundary
OVERLAP_S: float = 0.5


@dataclass
class ChunkResult:
    """Result of transcribing one streaming chunk."""

    index: int
    text: str
    segments: list[Segment] = field(default_factory=list)
    start_s: float = 0.0  # position in the full recording
    end_s: float = 0.0
    latency_ms: float = 0.0
    device: str = "cuda"
    language: str = "en"
    language_probability: float = 1.0
    is_final: bool = False


class StreamingTranscriber:
    """Transcribes audio in real time by monitoring an active AudioRecorder.

    Usage:
        streamer = StreamingTranscriber(
            recorder=recorder,
            config=config,
            on_chunk=lambda r: print(r.text),
            on_error=lambda e: print(e),
        )
        streamer.start()
        # ... recording happens ...
        streamer.stop()   # blocks until final chunk is transcribed
    """

    def __init__(
        self,
        recorder: AudioRecorder,
        config: Config,
        on_chunk: Callable[[ChunkResult], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._recorder = recorder
        self._config = config
        self._on_chunk = on_chunk
        self._on_error = on_error

        self._thread: threading.Thread | None = None
        self._chunk_index = 0
        self._consumed_samples = 0
        self._prev_context: str = ""
        self._prev_overlap_text: str = ""  # Text from previous chunk's overlap region
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background monitoring thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="stream-vad")
        self._thread.start()
        logger.debug("StreamingTranscriber started")

    def stop(self) -> None:
        """Signal stop and block until the final chunk is transcribed."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
        logger.debug("StreamingTranscriber stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_audio_snapshot(self) -> np.ndarray:
        """Thread-safe snapshot of all audio captured so far."""
        with self._recorder._lock:
            frames = list(self._recorder._frames)
        if not frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(frames).flatten()

    def _monitor_loop(self) -> None:
        """Background thread: poll audio, detect silence, dispatch chunks."""
        silence_start: float | None = None

        while not self._stop_event.is_set():
            time.sleep(POLL_INTERVAL_S)

            audio = self._get_audio_snapshot()
            pending_s = (len(audio) - self._consumed_samples) / SAMPLE_RATE

            if pending_s < MIN_CHUNK_S:
                continue

            # Check if the tail of the buffer is silent
            tail_samples = int(SILENCE_TRIGGER_S * SAMPLE_RATE)
            tail = audio[-tail_samples:] if len(audio) >= tail_samples else audio
            rms = float(np.sqrt(np.mean(tail**2)))

            if rms < SILENCE_RMS_THRESHOLD:
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= SILENCE_TRIGGER_S:
                    self._dispatch_chunk(audio, is_final=False)
                    silence_start = None
            else:
                silence_start = None

        # Recording stopped — transcribe whatever remains
        audio = self._get_audio_snapshot()
        remaining_s = (len(audio) - self._consumed_samples) / SAMPLE_RATE

        if remaining_s > 0.5 or self._chunk_index == 0:
            self._dispatch_chunk(audio, is_final=True)
        else:
            # Everything already dispatched — fire empty final signal
            self._on_chunk(
                ChunkResult(
                    index=self._chunk_index + 1,
                    text="",
                    start_s=self._consumed_samples / SAMPLE_RATE,
                    end_s=len(audio) / SAMPLE_RATE,
                    is_final=True,
                )
            )

    def _dispatch_chunk(self, full_audio: np.ndarray, is_final: bool) -> None:
        """Transcribe the pending audio and fire on_chunk."""
        from voicepad_core.transcription import (
            _DISTIL_MODELS,
            BEAM_SIZE,
            HALLUCINATION_SILENCE_THRESHOLD,
            INITIAL_PROMPT,
            LANGUAGE,
            AudioTooShortError,
            Segment,
            TranscriptionError,
            get_or_load_model,
        )

        # Include a short overlap for acoustic context at the boundary
        overlap_samples = int(OVERLAP_S * SAMPLE_RATE)
        start_sample = max(0, self._consumed_samples - overlap_samples)
        chunk_audio = full_audio[start_sample:]

        chunk_start_s = self._consumed_samples / SAMPLE_RATE
        chunk_end_s = len(full_audio) / SAMPLE_RATE
        audio_offset_s = start_sample / SAMPLE_RATE  # for adjusting segment timestamps

        logger.debug(
            f"Chunk {self._chunk_index + 1}: "
            f"{chunk_start_s:.1f}s–{chunk_end_s:.1f}s  "
            f"({len(chunk_audio) / SAMPLE_RATE:.1f}s incl. {OVERLAP_S}s overlap)"
        )

        try:
            is_distil = self._config.transcription_model in _DISTIL_MODELS
            # Disable condition_on_previous_text to prevent hallucinations/repeats.
            # VAD already provides natural chunk boundaries. We use context-aware
            # prompting instead to maintain continuity without amplifying errors.
            condition_on_prev = False
            if is_distil:
                prompt = None
            elif self._prev_context:
                # Include previous context in prompt for continuity
                prompt = (INITIAL_PROMPT + " " + self._prev_context[-200:]).strip()
            else:
                prompt = INITIAL_PROMPT

            t0 = time.perf_counter()
            model, device, _compute, _ = get_or_load_model(self._config)
            # Trim trailing silence to prevent tail hallucinations
            from voicepad_core.transcription import NO_SPEECH_THRESHOLD, _trim_trailing_silence

            chunk_audio = _trim_trailing_silence(chunk_audio)

            # VAD parameters tuned for better chunking within streaming chunks
            vad_params = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": float("inf"),
                "min_silence_duration_ms": 1000,  # 1s silence to split (vs default 2s)
                "speech_pad_ms": 400,
            }
            segs_iter, info = model.transcribe(
                chunk_audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters=vad_params,
                hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                initial_prompt=prompt,
                condition_on_previous_text=condition_on_prev,
            )
            raw_segments = list(segs_iter)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Adjust timestamps and verify overlap consistency
            segments: list[Segment] = []
            overlap_segments: list[Segment] = []  # Segments in the overlap region

            for s in raw_segments:
                abs_start = s.start + audio_offset_s
                abs_end = s.end + audio_offset_s

                # Categorize segments: overlap vs new content
                if abs_end <= chunk_start_s + 0.1:
                    # Segment entirely in overlap region
                    overlap_segments.append(Segment(start=abs_start, end=abs_end, text=s.text.strip()))
                elif abs_start < chunk_start_s and abs_end > chunk_start_s + 0.1:
                    # Segment spans the boundary - keep it to avoid word-splitting
                    segments.append(Segment(start=abs_start, end=abs_end, text=s.text.strip()))
                else:
                    # Segment entirely after overlap - keep it
                    segments.append(Segment(start=abs_start, end=abs_end, text=s.text.strip()))

            # Verify overlap consistency with previous chunk
            overlap_text = " ".join(s.text for s in overlap_segments if s.text).strip()
            if self._chunk_index > 0 and self._prev_overlap_text and overlap_text:
                # Normalize for comparison (remove extra spaces, lowercase)
                prev_normalized = " ".join(self._prev_overlap_text.lower().split())
                curr_normalized = " ".join(overlap_text.lower().split())

                # Check if overlap texts are similar (allow minor differences due to VAD)
                if prev_normalized and curr_normalized:
                    # Use simple substring check - current should contain most of previous
                    similarity = len(set(prev_normalized.split()) & set(curr_normalized.split()))
                    total_words = len(set(prev_normalized.split()))
                    if total_words > 0 and similarity / total_words < 0.5:
                        logger.warning(
                            f"Chunk {self._chunk_index + 1}: Overlap mismatch detected. "
                            f"Previous: '{self._prev_overlap_text[:50]}...' "
                            f"Current: '{overlap_text[:50]}...'"
                        )

            text = " ".join(s.text for s in segments if s.text).strip()

            # Store overlap text from end of current chunk for next verification
            # Get last OVERLAP_S seconds of segments for next chunk's verification
            overlap_boundary = chunk_end_s - OVERLAP_S
            end_segments = [s for s in segments if s.start >= overlap_boundary]
            self._prev_overlap_text = " ".join(s.text for s in end_segments if s.text).strip()

            # Update context for next chunk's prompt
            if text:
                self._prev_context = " ".join(text.split()[-30:])

            self._consumed_samples = len(full_audio)
            self._chunk_index += 1

            self._on_chunk(
                ChunkResult(
                    index=self._chunk_index,
                    text=text,
                    segments=segments,
                    start_s=chunk_start_s,
                    end_s=chunk_end_s,
                    latency_ms=latency_ms,
                    device=device,
                    language=info.language,
                    language_probability=info.language_probability,
                    is_final=is_final,
                )
            )

        except AudioTooShortError:
            self._consumed_samples = len(full_audio)
            if is_final:
                self._on_chunk(
                    ChunkResult(
                        index=self._chunk_index + 1,
                        text="",
                        start_s=chunk_start_s,
                        end_s=chunk_end_s,
                        is_final=True,
                    )
                )
        except (TranscriptionError, Exception) as e:
            logger.error(f"Chunk {self._chunk_index + 1} failed: {e}")
            self._on_error(str(e))
