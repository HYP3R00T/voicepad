from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from voicepad_core.audio import FileSource, RawAudio
from voicepad_core.inference import (
    CancellationToken,
    ResidentTranscriptionEngine,
    TranscriptionIntent,
)
from voicepad_core.planning import AdaptiveChunkPlanner
from voicepad_core.preprocessing import DEFAULT_WAVEFORM_SPEC, AudioPreProcessor, PreprocessedAudio
from voicepad_core.vad import PauseTracker, SileroVad, material_speech_regions

from .contracts import ReadyEngine, SequentialVad
from .transcript_assembly import ConservativeAssembler
from .types import ChunkOutcome, TranscriptionResult

logger = logging.getLogger(__name__)


class BatchTranscriber:
    """Run immutable audio through VAD, bounded inference, and one assembler."""

    def __init__(self, engine: ReadyEngine, vad: SequentialVad) -> None:
        self._engine = engine
        self._vad = vad

    def transcribe_file(
        self,
        path: Path,
        *,
        intent: TranscriptionIntent | None = None,
        cancellation: CancellationToken | None = None,
    ) -> TranscriptionResult:
        logger.info("Batch transcription file requested: path=%s", path)
        return self.transcribe_audio(
            FileSource(path).read_audio(),
            intent=intent,
            cancellation=cancellation,
        )

    def transcribe_audio(
        self,
        raw_audio: RawAudio,
        *,
        intent: TranscriptionIntent | None = None,
        cancellation: CancellationToken | None = None,
    ) -> TranscriptionResult:
        active = self._engine.active_deployment
        if active is None:
            raise RuntimeError("A verified and warmed transcription deployment must be active.")
        started = time.perf_counter()
        logger.info(
            "Batch transcription started: sample_rate=%s channels=%s frames=%s deployment=%s",
            raw_audio.sample_rate,
            raw_audio.channels,
            raw_audio.frame_count,
            active.definition.id,
        )
        requested_intent = intent or TranscriptionIntent()
        token = cancellation or CancellationToken()
        prepared = AudioPreProcessor.prepare(raw_audio, DEFAULT_WAVEFORM_SPEC)
        self._vad.reset()
        frames = self._vad.accept(prepared.samples, 0, final=True)
        tracker = PauseTracker()
        planner = AdaptiveChunkPlanner()
        for frame in frames:
            pause = tracker.accept(frame)
            if pause is not None:
                planner.add_pause(pause)
        descriptors = planner.poll(len(prepared.samples), final=True)
        speech = material_speech_regions(frames)
        assembler = ConservativeAssembler()
        outcomes: list[ChunkOutcome] = []
        warnings: list[str] = []

        for index, descriptor in enumerate(descriptors):
            if token.is_cancelled:
                warnings.append("job cancelled before all planned chunks were processed")
                break
            samples = np.ascontiguousarray(
                prepared.samples[descriptor.source_start_sample : descriptor.source_end_sample]
            )
            chunk_audio = PreprocessedAudio(samples, prepared.sample_rate, channels=1)
            chunk_started = time.perf_counter()
            try:
                result = self._engine.transcribe(chunk_audio, requested_intent, token)
                latency = time.perf_counter() - chunk_started
                assembler.add(index, descriptor, result)
                outcomes.append(
                    ChunkOutcome(
                        index,
                        descriptor,
                        latency,
                        len(result.tokens),
                        len(result.words),
                        result.cancelled,
                    )
                )
                logger.info(
                    "Batch transcription chunk completed: chunk=%s source_start=%s source_end=%s "
                    "latency_s=%.3f tokens=%s words=%s cancelled=%s",
                    index,
                    descriptor.source_start_sample,
                    descriptor.source_end_sample,
                    latency,
                    len(result.tokens),
                    len(result.words),
                    result.cancelled,
                )
                if result.cancelled:
                    warnings.append(f"chunk {index} generation was cancelled")
                    break
            except Exception as error:
                latency = time.perf_counter() - chunk_started
                message = f"{type(error).__name__}: {error}"
                outcomes.append(ChunkOutcome(index, descriptor, latency, 0, 0, False, message))
                warnings.append(f"chunk {index} failed")
                logger.error(
                    "Batch transcription chunk failed: chunk=%s error_type=%s error=%s",
                    index,
                    type(error).__name__,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )
                break

        gaps = assembler.coverage_gaps(speech)
        warnings.extend(assembler.warnings)
        if gaps:
            warnings.append(f"{len(gaps)} VAD-confirmed speech regions lack plausible timed-word coverage")
        all_chunks_terminal = len(outcomes) == len(descriptors) and all(
            outcome.error is None and not outcome.cancelled for outcome in outcomes
        )
        complete = all_chunks_terminal and assembler.protocol_valid and not gaps and not token.is_cancelled
        result = TranscriptionResult(
            text=assembler.text,
            words=assembler.words,
            tokens=assembler.tokens,
            duration_seconds=prepared.duration(),
            latency_seconds=time.perf_counter() - started,
            deployment=active,
            chunks=tuple(outcomes),
            speech_regions=speech,
            coverage_gaps=gaps,
            warnings=tuple(warnings),
            complete=complete,
        )
        logger.info(
            "Batch transcription finished: complete=%s chunks=%s duration_s=%.3f latency_s=%.3f "
            "transformations=%s warnings=%s",
            result.complete,
            len(result.chunks),
            result.duration_seconds,
            result.latency_seconds,
            prepared.transformations,
            len(result.warnings),
        )
        return result


def build_batch_transcriber(
    engine: ResidentTranscriptionEngine,
    silero_model: Path,
) -> BatchTranscriber:
    return BatchTranscriber(engine, SileroVad(silero_model))
