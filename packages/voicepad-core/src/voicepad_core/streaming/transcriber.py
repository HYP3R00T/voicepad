# streaming/transcriber.py

"""VAD-triggered streaming transcriber.

Monitors a live MicrophoneStream buffer on a background thread, detects
silence boundaries using Silero VAD, and dispatches audio chunks to the
inference engine during recording. Each transcribed chunk is delivered to
the caller via the on_chunk callback.

Architecture:
    MicrophoneStream (audio/)
        └─ StreamingTranscriber._monitor_loop()   [background thread]
              ├─ Silero VAD speech detection (vad/)
              ├─ _dispatch_chunk()
              │     ├─ inference.transcribe()      [engine.py]
              │     ├─ postprocessing pipeline
              │     └─ on_chunk(ChunkResult)       [caller callback]
              └─ stop() → final chunk dispatch

Public API:
    StreamingTranscriber(recorder, on_chunk, on_error, ...)
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

from .errors import StreamingConfigurationError
from .types import ChunkResult
from ..config import Config, get_config
from ..inference.constants import SAMPLE_RATE
from ..inference.errors import AudioTooShortError, TranscriptionError
from ..models import is_distil_model
from ..postprocessing import deduplicate_overlap, normalize, remove_hallucinations
from ..preprocessing import AudioPreProcessor
from ..vad import SileroVAD

logger = logging.getLogger(__name__)

_session_logger: logging.Logger | None = None


def set_streaming_session_logger(session_logger: logging.Logger | None) -> None:
    """Set the session logger for detailed streaming transcription logging.

    Args:
        session_logger: Logger instance for the current transcription session
    """
    global _session_logger
    _session_logger = session_logger


class StreamingTranscriber:
    """Real-time transcription by monitoring a live MicrophoneStream buffer.

    Polls the recorder's audio buffer at POLL_INTERVAL_S intervals.
    When min_chunk_s of audio has accumulated AND Silero VAD confirms
    silence lasting silence_threshold_ms, the chunk is dispatched to the
    inference engine and the result is delivered via on_chunk.

    A hard cap of max_chunk_s forces a split even without silence. This is
    a per-chunk safety boundary, not a session-length limit, and it keeps
    long recordings compatible with Whisper's 30s context window.

    Each chunk includes overlap_s of audio from the previous chunk's
    tail to preserve acoustic context at boundaries. The overlap region
    is deduplicated in post-processing so text is never doubled.

    Args:
        recorder:              An active MicrophoneStream instance. Must already
                               be started before calling StreamingTranscriber.start().
        on_chunk:              Callback invoked with a ChunkResult for every
                               transcribed chunk, including the final one.
        on_error:              Callback invoked with an error message string if
                               a chunk fails with an unexpected exception.
        model_name:            Whisper model to use. Defaults to DEFAULT_MODEL.
        device:                Inference device ('cuda' or 'cpu').
        compute_type:          CTranslate2 precision string.
        min_chunk_s:           Minimum audio before considering a split.
        max_chunk_s:           Per-chunk safety limit; sessions remain unbounded.
        overlap_s:             Cross-chunk audio overlap.
        silence_threshold_ms:  VAD silence duration to trigger a split.
    """

    def __init__(
        self,
        recorder,
        on_chunk: Callable[[ChunkResult], None],
        on_error: Callable[[str], None],
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        min_chunk_s: float | None = None,
        max_chunk_s: float | None = None,
        overlap_s: float | None = None,
        silence_threshold_ms: int | None = None,
        beam_size: int | None = None,
        vad_filter: bool | None = None,
        config: Config | None = None,
    ) -> None:
        self._config = config or get_config()
        self._recorder = recorder
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._model_name = model_name if model_name is not None else self._config.transcription_model
        self._device = device if device is not None else self._config.transcription_device
        self._compute_type = compute_type if compute_type is not None else self._config.transcription_compute_type

        self._min_chunk_s = min_chunk_s if min_chunk_s is not None else self._config.min_chunk_s
        self._max_chunk_s = max_chunk_s if max_chunk_s is not None else self._config.max_chunk_s
        self._overlap_s = overlap_s if overlap_s is not None else self._config.overlap_s
        self._silence_threshold_ms = (
            silence_threshold_ms if silence_threshold_ms is not None else self._config.silence_threshold_ms
        )
        self._beam_size = beam_size if beam_size is not None else self._config.beam_size
        self._vad_filter = vad_filter if vad_filter is not None else self._config.transcription_vad_filter
        self._poll_interval_s = self._config.stream_poll_interval_s
        self._validate_configuration()
        self._stream_context_chars = self._config.stream_context_chars

        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        self._consumed_samples: int = 0
        self._chunk_index: int = 0

        self._prev_context: str = ""
        self._prev_chunk_text: str = ""

        self._vad: SileroVAD | None = None

    def _validate_configuration(self) -> None:
        if self._min_chunk_s <= 0 or self._max_chunk_s <= 0:
            raise StreamingConfigurationError("min_chunk_s and max_chunk_s must be positive")
        if self._min_chunk_s > self._max_chunk_s:
            raise StreamingConfigurationError("min_chunk_s cannot be greater than max_chunk_s")
        if self._overlap_s < 0:
            raise StreamingConfigurationError("overlap_s cannot be negative")
        if self._silence_threshold_ms <= 0:
            raise StreamingConfigurationError("silence_threshold_ms must be positive")

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
        self._prev_context = ""
        self._prev_chunk_text = ""

        # Initialise VAD on first use (downloads ONNX model if needed)
        if self._vad is None:
            self._vad = SileroVAD(
                threshold=self._config.vad_threshold,
                min_speech_duration_ms=self._config.vad_min_speech_duration_ms,
                min_silence_duration_ms=self._silence_threshold_ms,
                speech_pad_ms=self._config.vad_speech_pad_ms,
                config=self._config,
            )

        self._vad.reset()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="stream-vad",
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

    def _monitor_loop(self) -> None:
        """Poll the recorder buffer and dispatch chunks on silence boundaries.

        Runs entirely on the background monitor thread. Uses Silero VAD to
        detect speech/silence instead of RMS energy thresholds. Dispatches
        one final chunk (is_final=True) before returning, even if no speech
        was found.
        """
        capture_rate: int = SAMPLE_RATE

        with contextlib.suppress(AttributeError):
            capture_rate = self._recorder.sample_rate

        while not self._stop_event.is_set():
            time.sleep(self._poll_interval_s)

            try:
                audio: np.ndarray = self._recorder.get_snapshot()
            except Exception as e:
                logger.error(f"StreamingTranscriber: failed to read audio buffer: {e}")
                continue

            if audio is None or len(audio) == 0:
                continue

            accumulated_s = (len(audio) - self._consumed_samples) / capture_rate

            if accumulated_s < self._min_chunk_s:
                if accumulated_s >= self._max_chunk_s:
                    logger.debug(f"Hard cap reached: {accumulated_s:.1f}s >= {self._max_chunk_s}s — forcing split.")
                    self._dispatch_chunk(audio, is_final=False, capture_rate=capture_rate)
                continue

            # Run VAD on the last silence_threshold_ms of audio.
            # If no speech is found in that window, silence is confirmed.
            tail_duration_s = self._silence_threshold_ms / 1000.0
            tail_samples = int(tail_duration_s * capture_rate)
            tail = audio[-tail_samples:] if len(audio) >= tail_samples else audio

            # VAD needs 16kHz — resample tail if needed
            if capture_rate != SAMPLE_RATE:
                tail = _resample(tail, capture_rate, SAMPLE_RATE)

            assert self._vad is not None
            speech_segments = self._vad.detect(tail, sample_rate=SAMPLE_RATE)

            if not speech_segments:
                logger.debug(f"VAD confirmed silence in last {tail_duration_s:.1f}s — dispatching chunk.")
                self._dispatch_chunk(audio, is_final=False, capture_rate=capture_rate)
                self._vad.reset()
            elif accumulated_s >= self._max_chunk_s:
                logger.debug(f"Hard cap reached: {accumulated_s:.1f}s >= {self._max_chunk_s}s — forcing split.")
                self._dispatch_chunk(audio, is_final=False, capture_rate=capture_rate)
                self._vad.reset()

        try:
            audio = self._recorder.get_snapshot()
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

    def _trim_final_audio_to_speech(
        self,
        full_audio: np.ndarray,
        fresh_audio: np.ndarray,
        capture_rate: int,
    ) -> np.ndarray:
        """Trim final audio to the last VAD-confirmed fresh speech boundary.

        Returns the original full audio when this is not a skippable tail.
        Returns audio truncated to the end of the last confirmed fresh speech.
        Returns audio truncated to the already-consumed boundary when there is
        no meaningful fresh speech to transcribe.
        """
        if len(fresh_audio) == 0 or self._vad is None:
            return full_audio[: self._consumed_samples]

        vad_audio = fresh_audio if capture_rate == SAMPLE_RATE else _resample(fresh_audio, capture_rate, SAMPLE_RATE)
        self._vad.reset()
        speech_segments = self._vad.detect(vad_audio, sample_rate=SAMPLE_RATE)
        if not speech_segments:
            return full_audio[: self._consumed_samples]

        speech_duration_s = sum(segment.end - segment.start for segment in speech_segments)
        if speech_duration_s < self._config.min_fresh_speech_duration_s:
            return full_audio[: self._consumed_samples]

        last_speech_end_s = min(max(segment.end for segment in speech_segments), len(vad_audio) / SAMPLE_RATE)
        fresh_end_samples = min(len(fresh_audio), int(round(last_speech_end_s * capture_rate)))
        return full_audio[: self._consumed_samples + fresh_end_samples]

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
        from ..inference import transcribe
        from ..inference.types import Segment

        slog = _session_logger

        overlap_samples = int(self._overlap_s * capture_rate)
        start_sample = max(0, self._consumed_samples - overlap_samples)
        fresh_audio = full_audio[self._consumed_samples :]
        final_audio = full_audio

        if is_final:
            final_audio = self._trim_final_audio_to_speech(full_audio, fresh_audio, capture_rate)
            fresh_audio = final_audio[self._consumed_samples :]

        chunk_audio = final_audio[start_sample:]

        chunk_start_s = self._consumed_samples / capture_rate
        chunk_end_s = len(final_audio) / capture_rate
        audio_offset_s = start_sample / capture_rate
        chunk_duration_s = len(chunk_audio) / capture_rate

        msg = (
            f"Chunk {self._chunk_index + 1}: "
            f"{chunk_start_s:.1f}s–{chunk_end_s:.1f}s "
            f"({chunk_duration_s:.1f}s incl. {self._overlap_s}s overlap)"
        )
        logger.debug(msg)
        if slog:
            slog.info(msg)
            slog.debug(
                f"  start_sample={start_sample}, consumed={self._consumed_samples}, "
                f"overlap_samples={overlap_samples}, chunk_samples={len(chunk_audio)}"
            )

        if slog and capture_rate != SAMPLE_RATE:
            slog.debug(f"Preprocessing chunk from {capture_rate}Hz to {SAMPLE_RATE}Hz")

        if is_final and len(fresh_audio) == 0:
            if slog:
                slog.info(f"Skipping final chunk {self._chunk_index + 1}: no fresh VAD-confirmed speech")
            self._consumed_samples = len(full_audio)
            self._on_chunk(
                ChunkResult(
                    index=self._chunk_index + 1,
                    text="",
                    start_s=chunk_start_s,
                    end_s=chunk_end_s,
                    is_final=True,
                )
            )
            return

        # Keep streaming chunks on the same preprocessing path as saved WAV
        # transcriptions so the final words are not lost to inconsistent audio prep.
        chunk_audio = AudioPreProcessor(self._recorder).process_array(chunk_audio, sample_rate=capture_rate)

        try:
            is_distil = is_distil_model(self._model_name)
            prompt = None if is_distil else _build_prompt(self._prev_context, self._config.initial_prompt)

            if slog:
                slog.debug(
                    f"Transcribing chunk {self._chunk_index + 1} with prompt: {prompt[:50] if prompt else '(none)'}..."
                )

            chunk_start_time = time.perf_counter()

            result = transcribe(
                chunk_audio,
                model_name=self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                initial_prompt=prompt,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
            )

            chunk_latency = (time.perf_counter() - chunk_start_time) * 1000

            if slog:
                slog.info(f"Chunk {self._chunk_index + 1} transcribed in {chunk_latency:.0f}ms")

            segments = []
            for s in result.segments:
                abs_start = s.start + audio_offset_s
                abs_end = s.end + audio_offset_s

                if abs_end <= chunk_start_s:
                    continue

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

            if slog:
                slog.debug(f"Reconstructed {len(segments)} segments with absolute timestamps")

            if self._prev_chunk_text and segments:
                if slog:
                    slog.debug("Deduplicating overlap with previous chunk")
                segments = deduplicate_overlap(
                    segments,
                    chunk_start_s,
                    self._prev_chunk_text,
                    prev_tail_words=self._config.dedup_prev_tail_words,
                    full_duplicate_threshold=self._config.dedup_full_duplicate_threshold,
                    min_overlap_words_for_partial=self._config.dedup_min_overlap_words_for_partial,
                    partial_lead_words=self._config.dedup_partial_lead_words,
                )

            text = " ".join(s.text for s in segments if s.text).strip()

            if slog:
                slog.debug(f"Raw text before post-processing: '{text[:100]}{'...' if len(text) > 100 else ''}'")

            if self._config.text_postprocessing_enabled:
                text = remove_hallucinations(text, max_repetitions=self._config.hallucination_max_repetitions)
                text = normalize(text)

            if slog:
                slog.info(
                    f"Chunk {self._chunk_index + 1} final text ({len(text)} chars): '{text[:100]}{'...' if len(text) > 100 else ''}'"
                )

            if text:
                self._prev_context = text[-self._stream_context_chars :]
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

        except AudioTooShortError as e:
            # emit an empty final marker if this was the last chunk
            if slog:
                slog.warning(f"Chunk {self._chunk_index + 1} too short: {e}")

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
            msg = f"Chunk {self._chunk_index + 1} failed: {e}"
            logger.error(msg)
            if slog:
                slog.error(msg)
            self._on_error(str(e))


def _build_prompt(prev_context: str, initial_prompt: str) -> str:
    """Build the Whisper initial prompt from previous context.

    Args:
        prev_context: Last 200 characters from the previous chunk.

    Returns:
        Combined prompt string.
    """
    if prev_context:
        return (initial_prompt + " " + prev_context).strip()
    return initial_prompt


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
