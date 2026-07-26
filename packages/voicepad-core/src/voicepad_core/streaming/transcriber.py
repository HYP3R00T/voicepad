from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .types import ChunkResult
from ..audio import AudioWindow, RawAudio
from ..config import Config, get_config
from ..inference.errors import AudioTooShortError
from ..inference.runtime import RuntimeManager
from ..inference.types import Segment, TranscriptionResult, WordTimestamp
from ..postprocessing import deduplicate_overlap, normalize, remove_hallucinations
from ..vad import SileroVAD
from ..vad.silero import REQUIRED_SAMPLE_RATE as VAD_SAMPLE_RATE

logger = logging.getLogger(__name__)

_session_logger: logging.Logger | None = None


class _Recorder(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def read_window(self, start_sample: int, max_samples: int) -> AudioWindow: ...


class StreamingConfigurationError(ValueError):
    """Raised when chunking settings are internally inconsistent."""


@dataclass(frozen=True, slots=True)
class _ChunkSlice:
    audio: np.ndarray
    end_sample: int
    start_s: float
    end_s: float
    audio_offset_s: float


def set_streaming_session_logger(session_logger: logging.Logger | None) -> None:
    """Set the logger for the current transcription session."""
    global _session_logger
    _session_logger = session_logger


class StreamingTranscriber:
    """Split a live microphone buffer on silence and transcribe each chunk."""

    def __init__(
        self,
        recorder: _Recorder,
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
        runtime_manager: RuntimeManager | None = None,
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
        self._runtime_manager = runtime_manager
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
        """Start monitoring; repeated calls are ignored."""
        if self._monitor_thread is not None:
            logger.warning("StreamingTranscriber.start() called more than once — ignored.")
            return

        self._stop_event.clear()
        self._consumed_samples = 0
        self._chunk_index = 0
        self._prev_context = ""
        self._prev_chunk_text = ""

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
        """Stop monitoring after emitting the final chunk."""
        if self._monitor_thread is None:
            return

        self._stop_event.set()
        self._monitor_thread.join(timeout=60.0)

        if self._monitor_thread.is_alive():
            logger.warning("StreamingTranscriber monitor thread did not finish within 60s.")

        self._monitor_thread = None
        logger.debug("StreamingTranscriber stopped.")

    def _monitor_loop(self) -> None:
        capture_rate = self._recorder.sample_rate

        while not self._stop_event.is_set():
            time.sleep(self._poll_interval_s)

            try:
                window = self._read_window(capture_rate)
            except Exception as exc:
                logger.error("Failed to read streaming audio buffer: %s", exc)
                continue

            if len(window.samples) == 0:
                continue

            accumulated_s = (window.end_sample - self._consumed_samples) / capture_rate

            if accumulated_s < self._min_chunk_s:
                continue

            # Run VAD on the last silence_threshold_ms of audio.
            # If no speech is found in that window, silence is confirmed.
            tail_duration_s = self._silence_threshold_ms / 1000.0
            tail_samples = int(tail_duration_s * capture_rate)
            tail = window.samples[-tail_samples:] if len(window.samples) >= tail_samples else window.samples

            # VAD needs 16kHz — resample tail if needed
            if capture_rate != VAD_SAMPLE_RATE:
                tail = _resample(tail, capture_rate, VAD_SAMPLE_RATE)

            assert self._vad is not None
            try:
                self._vad.reset()
                speech_segments = self._vad.detect(tail, sample_rate=VAD_SAMPLE_RATE)
            except Exception as exc:
                logger.error("StreamingTranscriber: VAD failed: %s", exc)
                self._on_error(str(exc))
                continue

            if not speech_segments:
                logger.debug("VAD confirmed silence in the last %.1fs; dispatching chunk.", tail_duration_s)
                self._dispatch_chunk(window, is_final=False, capture_rate=capture_rate)
                self._vad.reset()
            elif accumulated_s >= self._max_chunk_s:
                logger.debug(
                    "Streaming hard cap reached: %.1fs >= %.1fs; forcing split.", accumulated_s, self._max_chunk_s
                )
                self._dispatch_chunk(window, is_final=False, capture_rate=capture_rate)
                self._vad.reset()

        self._drain_final_chunks(capture_rate)

    def _drain_final_chunks(self, capture_rate: int) -> None:
        """Drain a persisted backlog without loading or inferring it all at once."""
        max_chunk_samples = int(self._max_chunk_s * capture_rate)
        while True:
            try:
                window = self._read_window(capture_rate)
            except Exception as exc:
                logger.error("StreamingTranscriber: failed to read final audio buffer: %s", exc)
                self._on_error(str(exc))
                self._emit_empty_final(
                    self._consumed_samples / capture_rate,
                    self._consumed_samples / capture_rate,
                )
                return

            fresh_samples = window.end_sample - self._consumed_samples
            if fresh_samples <= 0:
                self._emit_empty_final(
                    self._consumed_samples / capture_rate,
                    window.end_sample / capture_rate,
                )
                return

            is_final = fresh_samples < max_chunk_samples
            self._dispatch_chunk(window, is_final=is_final, capture_rate=capture_rate)
            if is_final:
                return

    def _read_window(self, capture_rate: int) -> AudioWindow:
        overlap_samples = int(self._overlap_s * capture_rate)
        start_sample = max(0, self._consumed_samples - overlap_samples)
        max_samples = overlap_samples + int(self._max_chunk_s * capture_rate)
        return self._recorder.read_window(start_sample, max_samples)

    def _trim_final_audio_to_speech(self, fresh_audio: np.ndarray, capture_rate: int) -> np.ndarray:
        """Trim the final tail to its last meaningful speech boundary."""
        if len(fresh_audio) == 0 or self._vad is None:
            return np.array([], dtype=np.float32)

        vad_audio = (
            fresh_audio if capture_rate == VAD_SAMPLE_RATE else _resample(fresh_audio, capture_rate, VAD_SAMPLE_RATE)
        )
        self._vad.reset()
        speech_segments = self._vad.detect(vad_audio, sample_rate=VAD_SAMPLE_RATE)
        if not speech_segments:
            return np.array([], dtype=np.float32)

        speech_duration_s = sum(segment.end - segment.start for segment in speech_segments)
        if speech_duration_s < self._config.min_fresh_speech_duration_s:
            return np.array([], dtype=np.float32)

        last_speech_end_s = min(max(segment.end for segment in speech_segments), len(vad_audio) / VAD_SAMPLE_RATE)
        fresh_end_samples = min(len(fresh_audio), int(round(last_speech_end_s * capture_rate)))
        return fresh_audio[:fresh_end_samples]

    def _dispatch_chunk(
        self,
        full_audio: np.ndarray | AudioWindow,
        is_final: bool,
        capture_rate: int = VAD_SAMPLE_RATE,
    ) -> None:
        """Prepare, transcribe, post-process, and emit one chunk."""
        from ..inference import transcribe

        slog = _session_logger
        window = full_audio if isinstance(full_audio, AudioWindow) else AudioWindow(full_audio, 0)
        chunk = self._prepare_chunk(window, is_final, capture_rate)
        if chunk is None:
            return

        try:
            prompt = _build_prompt(self._prev_context, self._config.initial_prompt)

            if slog:
                slog.debug(
                    "Transcribing chunk %s with prompt: %s...",
                    self._chunk_index + 1,
                    prompt[:50] if prompt else "(none)",
                )

            chunk_start_time = time.perf_counter()

            result = transcribe(
                RawAudio(chunk.audio, sample_rate=capture_rate, channels=1),
                model_name=self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                initial_prompt=prompt,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
                config=self._config.model_copy(update={"text_postprocessing_enabled": False}),
                runtime_manager=self._runtime_manager,
            )

            chunk_latency = (time.perf_counter() - chunk_start_time) * 1000

            if slog:
                slog.info("Chunk %s transcribed in %.0fms", self._chunk_index + 1, chunk_latency)

            self._emit_result(
                result,
                full_audio_length=window.end_sample if is_final else chunk.end_sample,
                chunk_start_s=chunk.start_s,
                chunk_end_s=chunk.end_s,
                audio_offset_s=chunk.audio_offset_s,
                is_final=is_final,
            )

        except AudioTooShortError as e:
            if slog:
                slog.warning("Chunk %s is too short: %s", self._chunk_index + 1, e)

            self._consumed_samples = window.end_sample if is_final else chunk.end_sample
            if is_final:
                self._emit_empty_final(chunk.start_s, chunk.end_s)

        except Exception as e:
            msg = f"Chunk {self._chunk_index + 1} failed: {e}"
            logger.error(msg)
            if slog:
                slog.error(msg)
            self._consumed_samples = window.end_sample if is_final else chunk.end_sample
            self._on_error(str(e))
            if is_final:
                self._emit_empty_final(chunk.start_s, chunk.end_s)

    def _prepare_chunk(
        self,
        window: AudioWindow,
        is_final: bool,
        capture_rate: int,
    ) -> _ChunkSlice | None:
        overlap_samples = int(self._overlap_s * capture_rate)
        start_sample = max(0, self._consumed_samples - overlap_samples)
        if start_sample < window.start_sample:
            raise StreamingConfigurationError(
                f"Recorder window starts at sample {window.start_sample}, but chunk requires {start_sample}."
            )
        fresh_offset = self._consumed_samples - window.start_sample
        fresh_audio = window.samples[fresh_offset:]
        if is_final:
            accepted_fresh = self._trim_final_audio_to_speech(fresh_audio, capture_rate)
        else:
            max_chunk_samples = int(self._max_chunk_s * capture_rate)
            accepted_fresh = fresh_audio[:max_chunk_samples]
        end_sample = self._consumed_samples + len(accepted_fresh)
        chunk_start_offset = start_sample - window.start_sample
        chunk_end_offset = end_sample - window.start_sample
        chunk_audio = window.samples[chunk_start_offset:chunk_end_offset]
        start_s = self._consumed_samples / capture_rate
        end_s = end_sample / capture_rate

        message = (
            f"Chunk {self._chunk_index + 1}: "
            f"{start_s:.1f}s–{end_s:.1f}s "
            f"({len(chunk_audio) / capture_rate:.1f}s incl. {self._overlap_s}s overlap)"
        )
        logger.debug(message)
        if _session_logger:
            _session_logger.info(message)
            _session_logger.debug(
                "start_sample=%s consumed=%s overlap_samples=%s chunk_samples=%s",
                start_sample,
                self._consumed_samples,
                overlap_samples,
                len(chunk_audio),
            )
            if capture_rate != VAD_SAMPLE_RATE:
                _session_logger.debug(
                    "Preprocessing chunk from %sHz to %sHz",
                    capture_rate,
                    VAD_SAMPLE_RATE,
                )

        if is_final and len(accepted_fresh) == 0:
            if _session_logger:
                _session_logger.info(
                    "Skipping final chunk %s: no fresh VAD-confirmed speech",
                    self._chunk_index + 1,
                )
            self._consumed_samples = window.end_sample
            self._emit_empty_final(start_s, end_s)
            return None

        return _ChunkSlice(
            audio=chunk_audio,
            end_sample=end_sample,
            start_s=start_s,
            end_s=end_s,
            audio_offset_s=start_sample / capture_rate,
        )

    def _emit_result(
        self,
        result: TranscriptionResult,
        *,
        full_audio_length: int,
        chunk_start_s: float,
        chunk_end_s: float,
        audio_offset_s: float,
        is_final: bool,
    ) -> None:
        segments = _absolute_segments(result.segments, audio_offset_s, chunk_start_s)
        slog = _session_logger

        if slog:
            slog.debug("Reconstructed %s segments with absolute timestamps", len(segments))

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

        text = " ".join(segment.text for segment in segments if segment.text).strip()
        if self._config.text_postprocessing_enabled:
            text = remove_hallucinations(
                text,
                max_repetitions=self._config.hallucination_max_repetitions,
            )
            text = normalize(text)
        if slog:
            slog.info(
                f"Chunk {self._chunk_index + 1} final text "
                f"({len(text)} chars): '{text[:100]}{'...' if len(text) > 100 else ''}'"
            )

        if text:
            self._prev_context = text[-self._stream_context_chars :]
            self._prev_chunk_text = text

        self._consumed_samples = full_audio_length
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

    def _emit_empty_final(self, start_s: float, end_s: float) -> None:
        self._on_chunk(
            ChunkResult(
                index=self._chunk_index + 1,
                text="",
                start_s=start_s,
                end_s=end_s,
                is_final=True,
            )
        )


def _build_prompt(prev_context: str, initial_prompt: str) -> str:
    if prev_context:
        return (initial_prompt + " " + prev_context).strip()
    return initial_prompt


def _absolute_segments(
    segments: list[Segment],
    audio_offset_s: float,
    chunk_start_s: float,
) -> list[Segment]:
    absolute: list[Segment] = []
    for segment in segments:
        abs_start = segment.start + audio_offset_s
        abs_end = segment.end + audio_offset_s
        if abs_end <= chunk_start_s:
            continue

        words = [
            WordTimestamp(
                word=word.word,
                start=word.start + audio_offset_s,
                end=word.end + audio_offset_s,
                probability=word.probability,
            )
            for word in segment.words
        ]
        absolute.append(
            Segment(
                start=abs_start,
                end=abs_end,
                text=segment.text,
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                words=words,
                confidence=getattr(segment, "confidence", None),
            )
        )
    return absolute


def _resample(
    audio: np.ndarray,
    orig_rate: int,
    target_rate: int,
) -> np.ndarray:
    if orig_rate == target_rate:
        return audio

    ratio = target_rate / orig_rate
    target_length = int(len(audio) * ratio)
    resampled = np.interp(
        np.linspace(0, len(audio) - 1, target_length),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)

    logger.debug(
        "Resampled streaming audio: %sHz -> %sHz (%s -> %s samples)",
        orig_rate,
        target_rate,
        len(audio),
        len(resampled),
    )
    return resampled
