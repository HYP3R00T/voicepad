from unittest.mock import MagicMock

from voicepad.runtime import ApplicationRuntime
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import FileTranscriptionResult


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
