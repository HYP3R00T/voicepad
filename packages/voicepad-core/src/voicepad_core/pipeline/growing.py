from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from voicepad_core.audio import GrowingAudioSource
from voicepad_core.inference import CancellationToken, TranscriptionIntent
from voicepad_core.planning import AdaptiveChunkPlanner, AudioChunk
from voicepad_core.preprocessing import PreprocessedAudio
from voicepad_core.vad import SAMPLE_RATE, PauseTracker, SileroVad, VadFrame, material_speech_regions

from .assembly import ConservativeAssembler
from .finite import ReadyEngine, SequentialVad
from .types import ChunkOutcome, FileTranscriptionResult, GrowingTranscriptionUpdate

_FINISHED = object()
VAD_READ_SAMPLES = SAMPLE_RATE
logger = logging.getLogger(__name__)


class GrowingPipelineError(RuntimeError):
    """Raised when a growing transcription job violates its lifecycle."""


class GrowingTranscriptionJob:
    """Transcribe a canonical growing disk source without blocking persistence."""

    def __init__(
        self,
        engine: ReadyEngine,
        vad: SequentialVad,
        source: GrowingAudioSource,
        *,
        intent: TranscriptionIntent | None = None,
        on_update: Callable[[GrowingTranscriptionUpdate], None] | None = None,
        descriptor_queue_size: int = 2,
    ) -> None:
        if source.sample_rate != SAMPLE_RATE or source.channels != 1:
            raise GrowingPipelineError("Growing transcription requires mono 16 kHz persisted audio.")
        if descriptor_queue_size <= 0:
            raise ValueError("descriptor queue size must be positive")
        active = engine.active_deployment
        if active is None:
            raise GrowingPipelineError("A verified and warmed transcription deployment must be active.")
        self._engine = engine
        self._vad = vad
        self._source = source
        self._intent = intent or TranscriptionIntent()
        self._on_update = on_update
        self._active = active
        self._queue: queue.Queue[AudioChunk | object] = queue.Queue(descriptor_queue_size)
        self._cancellation = CancellationToken()
        self._assembler = ConservativeAssembler()
        self._outcomes: list[ChunkOutcome] = []
        self._frames: list[VadFrame] = []
        self._warnings: list[str] = []
        self._planned_count = 0
        self._error: Exception | None = None
        self._started_at = 0.0
        self._planner_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._done = threading.Event()
        self._result: FileTranscriptionResult | None = None

    def start(self) -> None:
        if self._planner_thread is not None:
            raise GrowingPipelineError("Growing transcription job is already started.")
        self._started_at = time.perf_counter()
        self._vad.reset()
        self._planner_thread = threading.Thread(target=self._plan, name="transcription-planner", daemon=True)
        self._inference_thread = threading.Thread(target=self._infer, name="transcription-inference", daemon=True)
        self._inference_thread.start()
        self._planner_thread.start()

    def cancel(self) -> None:
        self._cancellation.cancel()

    def finish(self, timeout: float = 120.0) -> FileTranscriptionResult:
        if self._planner_thread is None:
            raise GrowingPipelineError("Growing transcription job is not started.")
        if not self._done.wait(timeout):
            raise GrowingPipelineError("Timed out while finalizing growing transcription.")
        if self._result is None:
            self._result = self._build_result()
        return self._result

    def _plan(self) -> None:
        planner = AdaptiveChunkPlanner()
        tracker = PauseTracker()
        cursor = 0
        try:
            while not self._cancellation.is_cancelled:
                committed, final = self._source.wait_for_update(cursor, timeout=0.1)
                while cursor < committed and not self._cancellation.is_cancelled:
                    end = min(cursor + VAD_READ_SAMPLES, committed)
                    window = self._source.read_range(cursor, end)
                    if window.start_sample != cursor or window.end_sample != end:
                        raise GrowingPipelineError("Growing source returned an incomplete committed range.")
                    is_terminal_read = final and end == committed
                    frames = self._vad.accept(window.samples, cursor, final=is_terminal_read)
                    self._frames.extend(frames)
                    for frame in frames:
                        pause = tracker.accept(frame)
                        if pause is not None:
                            planner.add_pause(pause)
                    cursor = end
                    for descriptor in planner.poll(cursor, final=is_terminal_read):
                        self._planned_count += 1
                        if not self._put_descriptor(descriptor):
                            break
                if final and cursor == committed:
                    for descriptor in planner.poll(cursor, final=True):
                        self._planned_count += 1
                        if not self._put_descriptor(descriptor):
                            break
                    break
        except Exception as error:
            self._error = error
            self._warnings.append("growing-source planning failed")
            self._cancellation.cancel()
        finally:
            self._put_finished()

    def _infer(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _FINISHED:
                        break
                    assert isinstance(item, AudioChunk)
                    if self._cancellation.is_cancelled:
                        continue
                    window = self._source.read_range(item.source_start_sample, item.source_end_sample)
                    if window.start_sample != item.source_start_sample or window.end_sample != item.source_end_sample:
                        raise GrowingPipelineError("Growing source lost a planned committed range.")
                    audio = PreprocessedAudio(np.ascontiguousarray(window.samples), SAMPLE_RATE, 1)
                    index = len(self._outcomes)
                    started = time.perf_counter()
                    try:
                        result = self._engine.transcribe(audio, self._intent, self._cancellation)
                        self._assembler.add(index, item, result)
                        self._outcomes.append(
                            ChunkOutcome(
                                index,
                                item,
                                time.perf_counter() - started,
                                len(result.tokens),
                                len(result.words),
                                result.cancelled,
                            )
                        )
                        self._publish_update(item)
                        if result.cancelled:
                            self._warnings.append(f"chunk {index} generation was cancelled")
                            self._cancellation.cancel()
                    except Exception as error:
                        self._outcomes.append(
                            ChunkOutcome(
                                index,
                                item,
                                time.perf_counter() - started,
                                0,
                                0,
                                False,
                                f"{type(error).__name__}: {error}",
                            )
                        )
                        self._error = error
                        self._warnings.append(f"chunk {index} failed")
                        self._cancellation.cancel()
                finally:
                    self._queue.task_done()
        finally:
            self._done.set()

    def _publish_update(self, descriptor: AudioChunk) -> None:
        if self._on_update is None:
            return
        update = GrowingTranscriptionUpdate(
            self._assembler.text,
            len(self._outcomes),
            descriptor.logical_end_sample,
        )
        try:
            self._on_update(update)
        except Exception as error:
            logger.warning("Growing transcription update callback failed: %s", error)

    def _put_descriptor(self, descriptor: AudioChunk) -> bool:
        while not self._cancellation.is_cancelled:
            try:
                self._queue.put(descriptor, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _put_finished(self) -> None:
        while True:
            try:
                self._queue.put(_FINISHED, timeout=0.1)
                return
            except queue.Full:
                continue

    def _build_result(self) -> FileTranscriptionResult:
        speech = material_speech_regions(tuple(self._frames))
        gaps = self._assembler.coverage_gaps(speech)
        warnings = [*self._warnings, *self._assembler.warnings]
        if gaps:
            warnings.append(f"{len(gaps)} VAD-confirmed speech regions lack plausible timed-word coverage")
        terminal = self._source.is_final
        all_chunks = len(self._outcomes) == self._planned_count and all(
            outcome.error is None and not outcome.cancelled for outcome in self._outcomes
        )
        complete = (
            terminal
            and all_chunks
            and self._error is None
            and not self._cancellation.is_cancelled
            and self._assembler.protocol_valid
            and not gaps
        )
        return FileTranscriptionResult(
            text=self._assembler.text,
            words=self._assembler.words,
            tokens=self._assembler.tokens,
            duration_seconds=self._source.committed_samples / SAMPLE_RATE,
            latency_seconds=time.perf_counter() - self._started_at,
            deployment=self._active,
            chunks=tuple(self._outcomes),
            speech_regions=speech,
            coverage_gaps=gaps,
            warnings=tuple(warnings),
            complete=complete,
        )


def build_growing_job(
    engine: ReadyEngine,
    silero_model: Path,
    source: GrowingAudioSource,
    *,
    intent: TranscriptionIntent | None = None,
    on_update: Callable[[GrowingTranscriptionUpdate], None] | None = None,
) -> GrowingTranscriptionJob:
    return GrowingTranscriptionJob(
        engine,
        SileroVad(silero_model),
        source,
        intent=intent,
        on_update=on_update,
    )
