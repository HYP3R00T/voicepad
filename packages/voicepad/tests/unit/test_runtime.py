from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from voicepad.config import AppConfig
from voicepad.runtime import ApplicationRuntime
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import FileTranscriptionResult


def test_recording_start_preserves_primary_failure_when_microphone_cleanup_fails(tmp_path: Path) -> None:
    runtime = ApplicationRuntime(AppConfig(recordings_path=tmp_path))
    runtime._silero_model = tmp_path / "silero.onnx"
    runtime.engine = MagicMock()
    microphone = MagicMock()
    microphone.stop.side_effect = RuntimeError("stop failed")
    with (
        patch("voicepad.runtime.MicrophoneStream", return_value=microphone),
        patch("voicepad.runtime.build_growing_job", side_effect=RuntimeError("job failed")),
        pytest.raises(RuntimeError, match="job failed") as failure,
    ):
        runtime.start_recording()

    assert failure.value.__notes__ == ["Microphone cleanup also failed: stop failed"]


def test_capture_error_marks_result_incomplete() -> None:
    source = PARAKEET_V3_MANIFEST.source
    assert isinstance(source, HuggingFaceSource)
    active = ActiveDeployment(PARAKEET_V3_CUDA, source.revision, "gpu", "gpu", 4_000_000_000)
    result = FileTranscriptionResult("partial", (), (), 1, 0.1, active, (), (), (), (), True)
    microphone = MagicMock()
    microphone.capture_error = RuntimeError("capture failed")
    job = MagicMock()
    job.finish.return_value = result

    _, finalized = ApplicationRuntime.stop_recording(microphone, job)

    assert finalized.complete is False
    assert finalized.warnings == ("audio capture failed: capture failed",)
