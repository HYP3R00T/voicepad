from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
from utilityhub_logging import LogFormat, begin_scope_logging, cleanup_logging, end_scope_logging
from voicepad_core.audio import AudioWindow, LiveWavRecording
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment, BackendResult, TimedWord, TokenTimestamp
from voicepad_core.pipeline import IncrementalTranscriptionJob, TranscriptionProgress
from voicepad_core.preprocessing import PreprocessedAudio
from voicepad_core.vad import VadFrame

SR = 16_000


class SpeechVad:
    def reset(self) -> None:
        pass

    def accept(self, samples: np.ndarray, start_sample: int, *, final: bool = False) -> tuple[VadFrame, ...]:
        return tuple(
            VadFrame(start_sample + offset, start_sample + min(offset + 512, len(samples)), 0.9)
            for offset in range(0, len(samples), 512)
        )


class BoundaryVad:
    def reset(self) -> None:
        pass

    def accept(self, samples: np.ndarray, start_sample: int, *, final: bool = False) -> tuple[VadFrame, ...]:
        frames = []
        for offset in range(0, len(samples), 512):
            absolute = start_sample + offset
            probability = 0.1 if 25 * SR <= absolute < 26 * SR else 0.9
            frames.append(VadFrame(absolute, start_sample + min(offset + 512, len(samples)), probability))
        return tuple(frames)


class FailingInferenceReadSource:
    sample_rate = SR
    channels = 1
    committed_samples = 2 * SR
    is_final = True

    def __init__(self) -> None:
        self.reads = 0

    def wait_for_update(self, after_sample: int, timeout: float | None = None) -> tuple[int, bool]:
        return self.committed_samples, True

    def read_range(self, start_sample: int, end_sample: int) -> AudioWindow:
        self.reads += 1
        if self.reads > 2:
            raise RuntimeError("inference read failed")
        return AudioWindow(np.zeros(end_sample - start_sample, dtype=np.float32), start_sample)


class FakeEngine:
    def __init__(self) -> None:
        source = PARAKEET_V3_MANIFEST.source
        assert isinstance(source, HuggingFaceSource)
        self.active_deployment = ActiveDeployment(
            PARAKEET_V3_CUDA,
            source.revision,
            "GPU-test",
            "NVIDIA Test GPU",
            4_000_000_000,
        )
        self.calls = 0

    def transcribe(self, audio: PreprocessedAudio, intent=None, cancellation=None) -> BackendResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        duration = audio.duration()
        word = TimedWord("incremental", 0.25, min(0.75, duration))
        return BackendResult(
            "incremental",
            (TokenTimestamp("▁incremental", word.start_seconds, word.end_seconds),),
            (word,),
        )


class PrivateTextEngine(FakeEngine):
    def transcribe(self, audio: PreprocessedAudio, intent=None, cancellation=None) -> BackendResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        word = TimedWord("PRIVATE_TRANSCRIPT_SENTINEL", 0.25, min(0.75, audio.duration()))
        return BackendResult(
            word.text,
            (TokenTimestamp("PRIVATE_TOKEN_SENTINEL", word.start_seconds, word.end_seconds),),
            (word,),
        )


def test_incremental_job_waits_for_persistence_finalization(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", SR, 1)
    recording.start()
    engine = FakeEngine()
    updates: list[TranscriptionProgress] = []
    job = IncrementalTranscriptionJob(engine, SpeechVad(), recording, on_update=updates.append)
    job.start()

    recording.append(np.zeros(SR, dtype=np.float32))
    recording.append(np.zeros(SR, dtype=np.float32))
    artifact = recording.finish()
    result = job.finish()

    assert artifact.path.exists()
    assert result.complete is True
    assert result.duration_seconds == 2.0
    assert result.text == "incremental"
    assert engine.calls == 1
    assert updates == [TranscriptionProgress("incremental", 1, 2 * SR)]


def test_incremental_job_emits_update_before_recording_stops(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", SR, 1)
    recording.start()
    received = Event()
    updates: list[TranscriptionProgress] = []

    def on_update(update: TranscriptionProgress) -> None:
        updates.append(update)
        received.set()

    job = IncrementalTranscriptionJob(FakeEngine(), BoundaryVad(), recording, on_update=on_update)
    job.start()
    recording.append(np.zeros(30 * SR, dtype=np.float32))

    assert received.wait(timeout=5)
    assert recording.is_final is False
    assert updates[0].processed_through_sample >= 25 * SR

    recording.finish()
    job.finish()


def test_incremental_job_cancellation_is_not_reported_complete(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", SR, 1)
    recording.start()
    job = IncrementalTranscriptionJob(FakeEngine(), SpeechVad(), recording)
    job.start()
    recording.append(np.zeros(SR, dtype=np.float32))

    job.cancel()
    recording.finish()
    result = job.finish()

    assert result.complete is False


def test_inference_read_failure_finishes_incomplete_without_leaking_workers() -> None:
    source = FailingInferenceReadSource()
    job = IncrementalTranscriptionJob(FakeEngine(), SpeechVad(), source)
    job.start()

    result = job.finish(timeout=2)

    assert result.complete is False
    assert result.chunks[0].error == "source read: RuntimeError: inference read failed"
    assert result.warnings[0] == "chunk 0 source read failed"
    assert job._planner_thread is not None and not job._planner_thread.is_alive()
    assert job._inference_thread is not None and not job._inference_thread.is_alive()


def test_unexpected_inference_worker_failure_does_not_strand_planner() -> None:
    source = FailingInferenceReadSource()
    job = IncrementalTranscriptionJob(FakeEngine(), SpeechVad(), source)
    with patch.object(job._queue, "get", side_effect=RuntimeError("queue failed")):
        job.start()
        result = job.finish(timeout=2)

    assert result.complete is False
    assert "incremental inference worker failed" in result.warnings
    assert job._planner_thread is not None and not job._planner_thread.is_alive()
    assert job._inference_thread is not None and not job._inference_thread.is_alive()


def test_incremental_source_publishes_committed_cursor_and_final_state(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", SR, 1)
    recording.start()
    recording.append(np.ones(512, dtype=np.float32))

    committed, final = recording.wait_for_update(0, timeout=2)
    window = recording.read_range(0, committed)

    assert committed == 512
    assert final is False
    assert window.end_sample == committed
    recording.finish()
    _, final = recording.wait_for_update(committed, timeout=2)
    assert final is True


def test_core_worker_logs_are_correlated_without_exposing_transcript_content(tmp_path: Path) -> None:
    scope_logger, log_path = begin_scope_logging(
        "recording",
        "core-worker-test",
        app_name="voicepad",
        logs_path=tmp_path,
        log_format=LogFormat.JSON,
    )
    context = {
        "recording_id": "core-worker-test",
        "scope_id": "core-worker-test",
        "scope_type": "recording",
        "component": "voicepad_core.pipeline.incremental",
    }
    scoped_logger = logging.LoggerAdapter(scope_logger.getChild("pipeline"), {"utilityhub_context": context})
    recording = LiveWavRecording(
        tmp_path / "recording.wav",
        SR,
        1,
        logger=scoped_logger,
        log_context=context,
    )
    job = IncrementalTranscriptionJob(
        PrivateTextEngine(),
        SpeechVad(),
        recording,
        logger=scoped_logger,
        log_context=context,
    )

    try:
        recording.start()
        job.start()
        recording.append(np.zeros(2 * SR, dtype=np.float32))
        recording.finish()
        result = job.finish()
    finally:
        end_scope_logging(scope_logger)
        cleanup_logging(close_all_loggers=True)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    chunk_records = [
        record for record in records if str(record["message"]).startswith("Incremental transcription chunk completed")
    ]
    assert result.text == "PRIVATE_TRANSCRIPT_SENTINEL"
    assert len(chunk_records) == 1
    assert chunk_records[0]["context"]["recording_id"] == "core-worker-test"
    assert chunk_records[0]["context"]["component"] == "voicepad_core.pipeline.incremental"
    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in log_path.read_text()
    assert "PRIVATE_TOKEN_SENTINEL" not in log_path.read_text()
