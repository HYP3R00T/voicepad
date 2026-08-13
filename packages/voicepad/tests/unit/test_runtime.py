from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from voicepad.config import AppConfig
from voicepad.runtime import ApplicationRuntime
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import TranscriptionResult


def test_recording_start_preserves_primary_failure_when_microphone_cleanup_fails(tmp_path: Path) -> None:
    runtime = ApplicationRuntime(AppConfig(recordings_path=tmp_path))
    runtime._silero_model = tmp_path / "silero.onnx"
    runtime.engine = MagicMock()
    microphone = MagicMock()
    microphone.stop.side_effect = RuntimeError("stop failed")
    with (
        patch("voicepad.runtime.RecordingLogScope.start") as start_scope,
        patch("voicepad.runtime.MicrophoneStream", return_value=microphone),
        patch("voicepad.runtime.build_incremental_job", side_effect=RuntimeError("job failed")),
        pytest.raises(RuntimeError, match="job failed") as failure,
    ):
        scope = start_scope.return_value
        scope.context = {"recording_id": "test"}
        runtime.start_recording()

    assert failure.value.__notes__ == ["Microphone cleanup also failed: stop failed"]
    scope.close.assert_called_once_with(outcome="failed", error=failure.value)
    assert runtime._recording_scope is None


def test_capture_error_marks_result_incomplete(tmp_path: Path) -> None:
    source = PARAKEET_V3_MANIFEST.source
    assert isinstance(source, HuggingFaceSource)
    active = ActiveDeployment(PARAKEET_V3_CUDA, source.revision, "gpu", "gpu", 4_000_000_000)
    result = TranscriptionResult("partial", (), (), 1, 0.1, active, (), (), (), (), True)
    microphone = MagicMock()
    microphone.capture_error = RuntimeError("capture failed")
    job = MagicMock()
    job.finish.return_value = result

    runtime = ApplicationRuntime(AppConfig(recordings_path=tmp_path))
    scope = MagicMock()
    runtime._recording_scope = scope

    _, finalized = runtime.stop_recording(microphone, job)

    assert finalized.complete is False
    assert finalized.warnings == ("audio capture failed: capture failed",)
    scope.close.assert_not_called()
    runtime.end_recording(outcome="incomplete")
    scope.close.assert_called_once_with(outcome="incomplete", error=None)
    assert runtime._recording_scope is None


def test_successful_recording_scope_stays_open_until_host_persistence_finishes(tmp_path: Path) -> None:
    source = PARAKEET_V3_MANIFEST.source
    assert isinstance(source, HuggingFaceSource)
    active = ActiveDeployment(PARAKEET_V3_CUDA, source.revision, "gpu", "gpu", 4_000_000_000)
    result = TranscriptionResult("complete", (), (), 1, 0.1, active, (), (), (), (), True)
    microphone = MagicMock(capture_error=None)
    job = MagicMock()
    job.finish.return_value = result
    runtime = ApplicationRuntime(AppConfig(recordings_path=tmp_path))
    scope = MagicMock()
    runtime._recording_scope = scope

    runtime.stop_recording(microphone, job)

    scope.close.assert_not_called()
    assert runtime._recording_scope is scope
    runtime.end_recording(outcome="completed")
    scope.close.assert_called_once_with(outcome="completed", error=None)


def test_recording_stop_failure_closes_scope_with_primary_error(tmp_path: Path) -> None:
    runtime = ApplicationRuntime(AppConfig(recordings_path=tmp_path))
    scope = MagicMock()
    runtime._recording_scope = scope
    microphone = MagicMock()
    microphone.stop.side_effect = RuntimeError("finalization failed")

    with pytest.raises(RuntimeError, match="finalization failed") as failure:
        runtime.stop_recording(microphone, MagicMock())

    scope.close.assert_called_once_with(outcome="failed", error=failure.value)
    assert runtime._recording_scope is None


def test_runtime_close_marks_unfinished_recording_abandoned(tmp_path: Path) -> None:
    runtime = ApplicationRuntime(AppConfig(recordings_path=tmp_path))
    runtime.engine = MagicMock()
    scope = MagicMock()
    runtime._recording_scope = scope

    runtime.close()

    runtime.engine.unload.assert_called_once_with()
    scope.close.assert_called_once_with(outcome="abandoned", error=None)
    assert runtime._recording_scope is None
