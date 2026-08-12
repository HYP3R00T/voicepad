from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from voicepad.runtime import ApplicationRuntime
from voicepad_core.audio import CaptureFailure, WavArtifact
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import FileTranscriptionResult


def _complete_result() -> FileTranscriptionResult:
    source = PARAKEET_V3_MANIFEST.source
    assert isinstance(source, HuggingFaceSource)
    active = ActiveDeployment(PARAKEET_V3_CUDA, source.revision, "GPU-test", "NVIDIA GPU", 4_000_000_000)
    return FileTranscriptionResult("partial words", (), (), 3.0, 0.1, active, (), (), (), (), True)


def test_capture_failure_marks_finalized_transcription_incomplete(tmp_path: Path) -> None:
    artifact = WavArtifact(tmp_path / "partial.wav", 16_000, 1, 48_000, 3.0)
    microphone = MagicMock()
    microphone.stop.return_value = artifact
    microphone.capture_failures = (CaptureFailure("capture-write", RuntimeError("durable writer failed")),)
    job = MagicMock()
    job.finish.return_value = _complete_result()

    saved, result = ApplicationRuntime.stop_recording(microphone, job)

    assert saved == artifact
    assert result.complete is False
    assert result.warnings == ("audio capture failed during capture-write: RuntimeError: durable writer failed",)
