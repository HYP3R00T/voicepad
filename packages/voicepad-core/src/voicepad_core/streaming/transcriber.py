# streaming/transcriber.py

"""VAD-triggered streaming transcriber.

Monitors a live AudioRecorder buffer on a background thread, detects
silence boundaries, and dispatches audio chunks to the inference engine
during recording. Each transcribed chunk is delivered to the caller via
the on_chunk callback.

Architecture:
    AudioRecorder (audio/)
        └─ StreamingTranscriber._monitor_loop()   [background thread]
              ├─ silence detection (RMS-based)
              ├─ _dispatch_chunk()
              │     ├─ inference.transcribe()      [engine.py]
              │     ├─ postprocessing pipeline
              │     └─ on_chunk(ChunkResult)       [caller callback]
              └─ stop() → final chunk dispatch

Public API:
    StreamingTranscriber(recorder, on_chunk, on_error, model_name, ...)
    .start()   → spawns monitor thread
    .stop()    → signals stop, waits for final chunk, joins thread
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from .chunk_result import ChunkResult
from ..inference.constants import COMPUTE_TYPE, DEFAULT_MODEL, DEVICE, SAMPLE_RATE
from ..inference.exceptions import AudioTooShortError, TranscriptionError
from ..postprocessing import deduplicate_overlap, normalize, remove_hallucinations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — same values as old codebase
# ---------------------------------------------------------------------------

# Minimum audio (seconds) accumulated before a silence-triggered split fires
MIN_CHUNK_S: float = 15.0

# RMS energy below this level is considered silence
SILENCE_RMS_THRESHOLD: float = 0.01

# How long silence must persist (seconds) to trigger a chunk dispatch
SILENCE_TRIGGER_S: float = 1.0

# How often the monitor thread polls the recorder buffer
POLL_INTERVAL_S: float = 0.3

# Audio overlap kept at chunk boundaries to preserve acoustic context
OVERLAP_S: float = 0.5


# ---------------------------------------------------------------------------
# StreamingTranscriber
# ---------------------------------------------------------------------------


class StreamingTranscriber:
    """Real-time transcription by monitoring a live AudioRecorder buffer.

    Polls the recorder's audio buffer at POLL_INTERVAL_S intervals.
    When MIN_CHUNK_S of audio has accumulated AND silence lasting
    SILENCE_TRIGGER_S is detected, the chunk is dispatched to the
    inference engine and the result is delivered via on_chunk.

    Each chunk includes OVERLAP_S of audio from the previous chunk's
    tail to preserve acoustic context at boundaries. The overlap region
    is deduplicated in post-processing so text is never doubled.

    Args:
        recorder:     An active AudioRecorder instance. Must already be
                      started before calling StreamingTranscriber.start().
        on_chunk:     Callback invoked with a ChunkResult for every
                      transcribed chunk, including the final one.
        on_error:     Callback invoked with an error message string if
                      a chunk fails with an unexpected exception.
        model_name:   Whisper model to use. Defaults to DEFAULT_MODEL.
        device:       Inference device ('cuda' or 'cpu').
        compute_type: CTranslate2 precision string.
    """

    def __init__(
        self,
        recorder,
        on_chunk: Callable[[ChunkResult], None],
        on_error: Callable[[str], None],
        model_name: str = DEFAULT_MODEL,
        device: str = DEVICE,
        compute_type: str = COMPUTE_TYPE,
    ) -> None:
        self._recorder = recorder
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type

        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        # Rolling state — reset on each start()
        self._consumed_samples: int = 0
        self._chunk_index: int = 0
        self._silence_since: float | None = None

        # Context carried across chunks
        self._prev_context: str = ""  # last 30 words for initial_prompt
        self._prev_chunk_text: str = ""  # full prev text for dedup

    # -----------------------------------------------------------------------
    # Public lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background monitor thread.

        Safe to call only once per instance. Create a new instance to
        restart a session.
        """
        if self._monitor_thread is not None:
            logger.warning("StreamingTranscriber.start() called more than once — ignored.")
            return

        self._stop_event.clear()
        self._consumed_samples = 0
        self._chunk_index = 0
        self._silence_since = None
        self._prev_context = ""
        self._prev_chunk_text = ""

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="streaming-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.debug("StreamingTranscriber monitor thread started.")

    def stop(self) -> None:
        """Signal the monitor thread to stop and wait for it to finish.

        After stop() returns, the final ChunkResult (is_final=True) has
        already been delivered via on_chunk. Safe to call multiple times.
        """
        if self._monitor_thread is None:
            return

        self._stop_event.set()
        self._monitor_thread.join(timeout=60.0)

        if self._monitor_thread.is_alive():
            logger.warning("StreamingTranscriber monitor thread did not finish within 60s.")

        self._monitor_thread = None
        logger.debug("StreamingTranscriber stopped.")

    # -----------------------------------------------------------------------
    # Monitor loop (background thread)
    # -----------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Poll the recorder buffer and dispatch chunks on silence boundaries.

        Runs entirely on the background monitor thread. Dispatches one final
        chunk (is_final=True) before returning, even if no speech was found.
        """
        capture_rate: int = SAMPLE_RATE

        # Attempt to read the recorder's native capture rate
        with contextlib.suppress(AttributeError):
            capture_rate = self._recorder.capture_rate

        while not self._stop_event.is_set():
            time.sleep(POLL_INTERVAL_S)

            try:
                audio: np.ndarray = self._recorder.get_audio()
            except Exception as e:
                logger.error(f"StreamingTranscriber: failed to read audio buffer: {e}")
                continue

            if audio is None or len(audio) == 0:
                continue

            accumulated_s = (len(audio) - self._consumed_samples) / capture_rate

            if accumulated_s < MIN_CHUNK_S:
                continue

            # --- Silence detection on the latest SILENCE_TRIGGER_S window ---
            window_samples = int(SILENCE_TRIGGER_S * capture_rate)
            tail = audio[-window_samples:] if len(audio) >= window_samples else audio
            rms = float(np.sqrt(np.mean(tail.astype(np.float32) ** 2)))
            is_silent = rms < SILENCE_RMS_THRESHOLD

            now = time.monotonic()

            if is_silent:
                if self._silence_since is None:
                    self._silence_since = now
                elif now - self._silence_since >= SILENCE_TRIGGER_S:
                    # Sustained silence — dispatch and reset silence timer
                    self._dispatch_chunk(audio, is_final=False, capture_rate=capture_rate)
                    self._silence_since = None
            else:
                self._silence_since = None

        # --- Final chunk: transcribe any remaining audio ---
        try:
            audio = self._recorder.get_audio()
        except Exception:
            audio = np.array([], dtype=np.float32)

        if audio is not None and self._consumed_samples < len(audio):
            self._dispatch_chunk(audio, is_final=True, capture_rate=capture_rate)
        else:
            # Nothing left to transcribe — emit an empty final marker
            self._on_chunk(
                ChunkResult(
                    index=self._chunk_index + 1,
                    text="",
                    start_s=self._consumed_samples / capture_rate,
                    end_s=len(audio) / capture_rate if audio is not None else 0.0,
                    is_final=True,
                )
            )

    # -----------------------------------------------------------------------
    # Chunk dispatch
    # -----------------------------------------------------------------------

    def _dispatch_chunk(
        self,
        full_audio: np.ndarray,
        is_final: bool,
        capture_rate: int = SAMPLE_RATE,
    ) -> None:
        """Slice, resample, transcribe, post-process and emit one chunk.

        Args:
            full_audio:   Complete audio buffer from the recorder (all samples
                          recorded so far, not just the new ones).
            is_final:     True when this is the last chunk of the session.
            capture_rate: Native sample rate of the recorder buffer.
        """
        # Import here to avoid circular imports at module load time
        from ..inference import transcribe

        overlap_samples = int(OVERLAP_S * capture_rate)
        start_sample = max(0, self._consumed_samples - overlap_samples)
        chunk_audio = full_audio[start_sample:]

        chunk_start_s = self._consumed_samples / capture_rate
        chunk_end_s = len(full_audio) / capture_rate
        audio_offset_s = start_sample / capture_rate

        logger.debug(
            f"Chunk {self._chunk_index + 1}: "
            f"{chunk_start_s:.1f}s–{chunk_end_s:.1f}s "
            f"({len(chunk_audio) / capture_rate:.1f}s incl. {OVERLAP_S}s overlap)"
        )

        # Resample to 16kHz if the recorder captures at a different rate
        if capture_rate != SAMPLE_RATE:
            chunk_audio = _resample(chunk_audio, capture_rate, SAMPLE_RATE)

        try:
            result = transcribe(
                chunk_audio,
                model_name=self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )

            # --- Reconstruct segments with absolute timestamps ---
            segments = []
            for s in result.segments:
                abs_start = s.start + audio_offset_s
                abs_end = s.end + audio_offset_s

                # Drop segments that end before the chunk's logical start
                if abs_end <= chunk_start_s:
                    continue

                from ..inference.types import Segment

                segments.append(
                    Segment(
                        start=abs_start,
                        end=abs_end,
                        text=s.text,
                        avg_logprob=s.avg_logprob,
                        no_speech_prob=s.no_speech_prob,
                        words=s.words,
                    )
                )

            # --- Post-processing pipeline ---
            if self._prev_chunk_text and segments:
                segments = deduplicate_overlap(segments, chunk_start_s, self._prev_chunk_text)

            text = " ".join(s.text for s in segments if s.text).strip()
            text = remove_hallucinations(text)
            text = normalize(text)

            # Update rolling context for next chunk
            if text:
                self._prev_context = " ".join(text.split()[-30:])
                self._prev_chunk_text = text

            self._consumed_samples = len(full_audio)
            self._chunk_index += 1

            self._on_chunk(
                ChunkResult(
                    index=self._chunk_index,
                    text=text,
                    segments=segments,
                    start_s=chunk_start_s,
                    end_s=chunk_end_s,
                    latency_ms=result.latency_ms,
                    device=result.device,
                    language=result.language,
                    language_probability=result.language_probability,
                    is_final=is_final,
                )
            )

        except AudioTooShortError:
            # Audio was too short to transcribe — advance pointer and
            # emit an empty final marker if this was the last chunk
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


# ---------------------------------------------------------------------------
# Internal — resampling helper
# ---------------------------------------------------------------------------


def _resample(
    audio: np.ndarray,
    orig_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample a float32 mono audio array from orig_rate to target_rate.

    Uses linear interpolation — fast and sufficient for 16kHz conversion.
    For production-quality resampling, replace with scipy.signal.resample
    or soxr if available.

    Args:
        audio:       float32 mono numpy array.
        orig_rate:   Sample rate of the input array.
        target_rate: Desired output sample rate.

    Returns:
        Resampled float32 numpy array at target_rate.
    """
    if orig_rate == target_rate:
        return audio

    ratio = target_rate / orig_rate
    target_length = int(len(audio) * ratio)
    resampled = np.interp(
        np.linspace(0, len(audio) - 1, target_length),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)

    logger.debug(f"_resample: {orig_rate}Hz → {target_rate}Hz ({len(audio)} → {len(resampled)} samples)")
    return resampled
