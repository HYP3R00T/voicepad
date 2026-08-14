from pathlib import Path

import numpy as np
import pytest
from voicepad_core.artifacts import ProgressCallback
from voicepad_core.deployments import (
    PARAKEET_V3_CUDA,
    PARAKEET_V3_MANIFEST,
    ArtifactManifest,
    HuggingFaceSource,
)
from voicepad_core.inference import (
    ActiveDeployment,
    BackendResult,
    CancellationToken,
    CudaDevice,
    InferenceError,
    ResidentTranscriptionEngine,
    TranscriptionIntent,
    UnsupportedIntentError,
)
from voicepad_core.preprocessing import PreprocessedAudio


class FakeStore:
    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def prepare(
        self,
        manifest: ArtifactManifest,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        self.calls += 1
        return self.snapshot


class FakeSession:
    def __init__(
        self,
        *,
        failure: Exception | None = None,
        warm_failure: Exception | None = None,
        close_failure: Exception | None = None,
    ) -> None:
        source = PARAKEET_V3_MANIFEST.source
        assert isinstance(source, HuggingFaceSource)
        self.deployment = ActiveDeployment(
            PARAKEET_V3_CUDA,
            source.revision,
            "GPU-test",
            "NVIDIA Test GPU",
            4_000_000_000,
        )
        self.failure = failure
        self.warm_failure = warm_failure
        self.close_failure = close_failure
        self.warmed = False
        self.closed = False

    def warm(self) -> None:
        self.warmed = True
        if self.warm_failure is not None:
            raise self.warm_failure

    def transcribe(
        self,
        audio: PreprocessedAudio,
        intent: TranscriptionIntent,
        cancellation: CancellationToken,
    ) -> BackendResult:
        if self.failure is not None:
            raise self.failure
        return BackendResult("ready", (), (), cancellation.is_cancelled)

    def close(self) -> None:
        self.closed = True
        if self.close_failure is not None:
            raise self.close_failure


def device() -> CudaDevice:
    return CudaDevice(0, "GPU-test", "NVIDIA Test GPU", 4_000_000_000, 3_000_000_000, (8, 6))


def audio() -> PreprocessedAudio:
    return PreprocessedAudio(np.zeros(16_000, dtype=np.float32), 16_000, 1)


def build_engine(tmp_path: Path, session: FakeSession) -> tuple[ResidentTranscriptionEngine, FakeStore]:
    store = FakeStore(tmp_path / "snapshot")
    engine = ResidentTranscriptionEngine(
        tmp_path / "artifacts",
        artifact_store=store,
        device_admitter=lambda definition, index: device(),
        session_factory=lambda *args: session,
    )
    return engine, store


def test_activate_warms_and_reuses_resident_session(tmp_path: Path) -> None:
    session = FakeSession()
    engine, store = build_engine(tmp_path, session)

    active = engine.activate(PARAKEET_V3_CUDA.id)
    repeated = engine.activate(PARAKEET_V3_CUDA.id)

    assert active == repeated
    assert engine.active_deployment == active
    assert session.warmed is True
    assert store.calls == 1


def test_transcribe_returns_to_ready_and_unload_closes_session(tmp_path: Path) -> None:
    session = FakeSession()
    engine, _ = build_engine(tmp_path, session)
    engine.activate(PARAKEET_V3_CUDA.id)

    result = engine.transcribe(audio())
    engine.unload()

    assert result.text == "ready"
    assert session.closed is True
    assert engine.active_deployment is None


def test_unsupported_intent_does_not_invalidate_session(tmp_path: Path) -> None:
    session = FakeSession(failure=UnsupportedIntentError("unsupported"))
    engine, _ = build_engine(tmp_path, session)
    engine.activate(PARAKEET_V3_CUDA.id)

    with pytest.raises(UnsupportedIntentError):
        engine.transcribe(audio())

    assert engine.active_deployment is not None
    assert session.closed is False


def test_activation_preserves_primary_failure_when_candidate_cleanup_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    session = FakeSession(
        warm_failure=InferenceError("warm failed"),
        close_failure=RuntimeError("close failed"),
    )
    engine, _ = build_engine(tmp_path, session)

    with pytest.raises(InferenceError, match="warm failed") as failure:
        engine.activate(PARAKEET_V3_CUDA.id)

    assert failure.value.__notes__ == ["Session cleanup also failed: close failed"]
    assert "Session cleanup failed after activation failure" in caplog.text
    assert engine.active_deployment is None
    engine.unload()
    assert engine.active_deployment is None


def test_unknown_inference_failure_invalidates_and_closes_session(tmp_path: Path) -> None:
    session = FakeSession(failure=InferenceError("CUDA failure"))
    engine, _ = build_engine(tmp_path, session)
    engine.activate(PARAKEET_V3_CUDA.id)

    with pytest.raises(InferenceError, match="CUDA failure"):
        engine.transcribe(audio())

    assert engine.active_deployment is None
    assert session.closed is True


def test_transcription_preserves_primary_failure_when_session_cleanup_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    session = FakeSession(
        failure=InferenceError("CUDA failure"),
        close_failure=RuntimeError("close failed"),
    )
    engine, _ = build_engine(tmp_path, session)
    engine.activate(PARAKEET_V3_CUDA.id)

    with pytest.raises(InferenceError, match="CUDA failure") as failure:
        engine.transcribe(audio())

    assert failure.value.__notes__ == ["Session cleanup also failed: close failed"]
    assert "Session cleanup failed after transcription failure" in caplog.text
    assert engine.active_deployment is None
    engine.unload()


def test_unload_failure_releases_operation_lock(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    session = FakeSession(close_failure=RuntimeError("close failed"))
    engine, _ = build_engine(tmp_path, session)
    engine.activate(PARAKEET_V3_CUDA.id)

    with pytest.raises(RuntimeError, match="close failed"):
        engine.unload()

    assert "Session cleanup failed while unloading" in caplog.text
    assert engine.active_deployment is None
    engine.unload()
    assert engine.active_deployment is None
