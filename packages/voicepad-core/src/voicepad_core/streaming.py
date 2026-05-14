"""VAD-triggered streaming transcription.

Polls AudioRecorder buffer, detects silence boundaries, and dispatches chunks
to Whisper during recording. Background thread accumulates audio until silence
detected after MIN_CHUNK_S, transcribes, and calls on_chunk callback.
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

MIN_CHUNK_S: float = 15.0  # Minimum audio before splitting (aligned with standard transcription)
SILENCE_RMS_THRESHOLD: float = 0.01  # RMS threshold for silence detection
SILENCE_TRIGGER_S: float = 1.0  # Silence duration to trigger split (aligned with VAD params)
POLL_INTERVAL_S: float = 0.3  # Monitor thread poll interval
OVERLAP_S: float = 0.5  # Audio overlap for acoustic context at boundaries


@dataclass
class ChunkResult:
    """Result of transcribing one streaming chunk."""

    index: int
    text: str
    segments: list[Segment] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0
    latency_ms: float = 0.0
    device: str = "cuda"
    language: str = "en"
    language_probability: float = 1.0
    is_final: bool = False


class StreamingTranscriber:
    """Real-time transcription by monitoring active AudioRecorder.

    Polls recorder buffer, detects silence boundaries, and dispatches chunks
    to Whisper. Calls on_chunk callback with results. stop() blocks until
    final chunk is transcribed.
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
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background monitoring thread for real-time transcription."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="stream-vad")
        self._thread.start()
        logger.debug("StreamingTranscriber started")

    def stop(self) -> None:
        """Stop monitoring and wait for final chunk transcription."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
        logger.debug("StreamingTranscriber stopped")

    def _get_audio_snapshot(self) -> np.ndarray:
        """Thread-safe snapshot of all captured audio."""
        with self._recorder._lock:
            frames = list(self._recorder._frames)
        if not frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(frames).flatten()

    @property
    def _capture_rate(self) -> int:
        """Recorder's actual sample rate."""
        return getattr(self._recorder, "_capture_rate", SAMPLE_RATE)

    def _monitor_loop(self) -> None:
        """Poll audio, detect silence, dispatch chunks."""
        silence_start: float | None = None
        capture_rate = self._capture_rate

        while not self._stop_event.is_set():
            time.sleep(POLL_INTERVAL_S)

            audio = self._get_audio_snapshot()
            pending_s = (len(audio) - self._consumed_samples) / capture_rate

            if pending_s < MIN_CHUNK_S:
                continue

            tail_samples = int(SILENCE_TRIGGER_S * capture_rate)
            tail = audio[-tail_samples:] if len(audio) >= tail_samples else audio
            rms = float(np.sqrt(np.mean(tail**2)))

            if rms < SILENCE_RMS_THRESHOLD:
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= SILENCE_TRIGGER_S:
                    self._dispatch_chunk(audio, is_final=False, capture_rate=capture_rate)
                    silence_start = None
            else:
                silence_start = None

        capture_rate = self._capture_rate
        audio = self._get_audio_snapshot()
        remaining_s = (len(audio) - self._consumed_samples) / capture_rate

        if remaining_s > 0.5 or self._chunk_index == 0:
            self._dispatch_chunk(audio, is_final=True, capture_rate=capture_rate)
        else:
            self._on_chunk(
                ChunkResult(
                    index=self._chunk_index + 1,
                    text="",
                    start_s=self._consumed_samples / capture_rate,
                    end_s=len(audio) / capture_rate,
                    is_final=True,
                )
            )

    def _dispatch_chunk(self, full_audio: np.ndarray, is_final: bool, capture_rate: int = SAMPLE_RATE) -> None:
        """Transcribe pending audio chunk and invoke callback.

        Args:
            full_audio: Complete audio buffer from recorder
            is_final: Whether this is the last chunk
            capture_rate: Sample rate of the audio buffer
        """
        from voicepad_core.audio import _resample
        from voicepad_core.transcription import (
            BEAM_SIZE,
            DISTIL_MODELS,
            HALLUCINATION_SILENCE_THRESHOLD,
            INITIAL_PROMPT,
            LANGUAGE,
            NO_SPEECH_THRESHOLD,
            AudioTooShortError,
            Segment,
            TranscriptionError,
            _get_vad_parameters,
            _trim_trailing_silence,
            get_or_load_model,
        )

        overlap_samples = int(OVERLAP_S * capture_rate)
        start_sample = max(0, self._consumed_samples - overlap_samples)
        chunk_audio = full_audio[start_sample:]

        chunk_start_s = self._consumed_samples / capture_rate
        chunk_end_s = len(full_audio) / capture_rate
        audio_offset_s = start_sample / capture_rate

        logger.debug(
            f"Chunk {self._chunk_index + 1}: "
            f"{chunk_start_s:.1f}s–{chunk_end_s:.1f}s  "
            f"({len(chunk_audio) / capture_rate:.1f}s incl. {OVERLAP_S}s overlap)"
        )

        if capture_rate != SAMPLE_RATE:
            chunk_audio = _resample(chunk_audio, capture_rate, SAMPLE_RATE)

        try:
            is_distil = self._config.transcription_model in DISTIL_MODELS
            if is_distil:
                prompt = None
            elif self._prev_context:
                prompt = (INITIAL_PROMPT + " " + self._prev_context[-200:]).strip()
            else:
                prompt = INITIAL_PROMPT

            t0 = time.perf_counter()
            model, device, _compute, _ = get_or_load_model(self._config)
            chunk_audio = _trim_trailing_silence(chunk_audio)

            vad_params = _get_vad_parameters()
            segs_iter, info = model.transcribe(
                chunk_audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                vad_parameters=vad_params,
                hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                initial_prompt=prompt,
                condition_on_previous_text=False,
            )
            raw_segments = list(segs_iter)
            latency_ms = (time.perf_counter() - t0) * 1000

            segments: list[Segment] = []
            for s in raw_segments:
                abs_start = s.start + audio_offset_s
                abs_end = s.end + audio_offset_s

                if abs_end <= chunk_start_s:
                    continue

                segments.append(
                    Segment(
                        start=abs_start,
                        end=abs_end,
                        text=s.text.strip(),
                        avg_logprob=s.avg_logprob,
                        no_speech_prob=s.no_speech_prob,
                    )
                )

            text = " ".join(s.text for s in segments if s.text).strip()

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
