from __future__ import annotations

from pathlib import Path

import numpy as np
from voicepad_core.audio import LiveWavRecording
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment, BackendResult, TimedWord, TokenTimestamp
from voicepad_core.pipeline import GrowingTranscriptionJob
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
        word = TimedWord("growing", 0.25, min(0.75, duration))
        return BackendResult("growing", (TokenTimestamp("▁growing", word.start_seconds, word.end_seconds),), (word,))


def test_growing_job_waits_for_persistence_finalization(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", SR, 1)
    recording.start()
    engine = FakeEngine()
    job = GrowingTranscriptionJob(engine, SpeechVad(), recording)
    job.start()

    recording.append(np.zeros(SR, dtype=np.float32))
    recording.append(np.zeros(SR, dtype=np.float32))
    artifact = recording.finish()
    result = job.finish()

    assert artifact.path.exists()
    assert result.complete is True
    assert result.duration_seconds == 2.0
    assert result.text == "growing"
    assert engine.calls == 1


def test_growing_job_cancellation_is_not_reported_complete(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", SR, 1)
    recording.start()
    job = GrowingTranscriptionJob(FakeEngine(), SpeechVad(), recording)
    job.start()
    recording.append(np.zeros(SR, dtype=np.float32))

    job.cancel()
    recording.finish()
    result = job.finish()

    assert result.complete is False


def test_growing_source_publishes_committed_cursor_and_final_state(tmp_path: Path) -> None:
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
