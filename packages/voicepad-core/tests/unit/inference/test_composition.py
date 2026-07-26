from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import voicepad_core.inference.composition as composition
from voicepad_core.inference.composition import (
    InferenceCoordinator,
    activate_model,
    deactivate_model,
    get_default_coordinator,
    model_is_ready,
    prepare_model,
)
from voicepad_core.inference.contracts import (
    BackendCapabilities,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from voicepad_core.inference.types import TranscriptionResult
from voicepad_core.models import LocalArtifact, ModelSpec


class _Session:
    def __init__(self, info: RuntimeInfo) -> None:
        self._info = info
        self.close_calls = 0

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        return TranscriptionResult(
            text="",
            segments=[],
            language="en",
            language_probability=1.0,
            duration_s=request.audio.size / request.sample_rate,
            latency_ms=0.0,
            device=self.info.device,
            compute_type=self.info.precision,
        )

    def close(self) -> None:
        self.close_calls += 1


class _Driver:
    def __init__(self, backend_id: str = "test") -> None:
        self._id = backend_id
        self.sessions: list[_Session] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(context_biasing=True)

    def is_available(self) -> bool:
        return True

    def prepare(self, model: ModelSpec) -> PreparedModel:
        return PreparedModel(model, Path(model.id))

    def open(self, model: PreparedModel, options: RuntimeOptions) -> _Session:
        session = _Session(
            RuntimeInfo(
                backend_id=self.id,
                model_id=model.spec.id,
                device=options.device,
                precision=options.precision,
            )
        )
        self.sessions.append(session)
        return session


class _FailingCoordinator:
    def deactivate(self) -> None:
        raise RuntimeError("close failed")


def _model(model_id: str, artifact_path: Path | None = None) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        backend_id="test",
        artifact_format="test",
        artifact_source=LocalArtifact(artifact_path or Path(model_id)),
        required_files=("weights.bin",),
    )


def _progress(downloaded: int, total: int | None) -> None:
    del downloaded, total


class TestInferenceCoordinator:
    def test_default_coordinator_registers_both_builtin_backends(self, tmp_path: Path) -> None:
        """A default coordinator enables Faster-Whisper and Parakeet in stable order."""
        coordinator = InferenceCoordinator(tmp_path)

        assert coordinator.backend_ids == ("faster-whisper", "parakeet-onnx")

    def test_prepare_validates_a_local_artifact(self, tmp_path: Path) -> None:
        """Preparing a declared local model returns its validated artifact directory."""
        artifact_path = tmp_path / "artifact"
        artifact_path.mkdir()
        (artifact_path / "weights.bin").touch()
        coordinator = InferenceCoordinator(tmp_path / "cache", drivers=())

        prepared_path = coordinator.prepare(_model("local", artifact_path))

        assert prepared_path == artifact_path.resolve()

    def test_is_prepared_does_not_acquire_missing_artifact(self, tmp_path: Path) -> None:
        """A missing local artifact reports not ready without creating the path."""
        artifact_path = tmp_path / "missing"
        coordinator = InferenceCoordinator(tmp_path / "cache", drivers=())

        prepared = coordinator.is_prepared(_model("local", artifact_path))

        assert (prepared, artifact_path.exists()) == (False, False)

    def test_activate_resolves_catalog_model_id(self, tmp_path: Path) -> None:
        """A string model id resolves through the catalogue before backend activation."""
        driver = _Driver("faster-whisper")
        coordinator = InferenceCoordinator(tmp_path, drivers=(driver,))

        session = coordinator.activate(
            "tiny",
            RuntimeOptions(device="cpu", precision="int8"),
        )

        assert session.info == RuntimeInfo("faster-whisper", "tiny", "cpu", "int8")

    def test_activate_replaces_the_only_resident_runtime(self, tmp_path: Path) -> None:
        """Activating another model closes the old session before exposing the new one."""
        driver = _Driver()
        coordinator = InferenceCoordinator(tmp_path, drivers=(driver,))
        coordinator.activate(
            _model("first"),
            RuntimeOptions(device="cpu", precision="int8"),
        )

        coordinator.activate(
            _model("second"),
            RuntimeOptions(device="cpu", precision="int8"),
        )

        assert (
            driver.sessions[0].close_calls,
            driver.sessions[1].close_calls,
            coordinator.active_info,
        ) == (
            1,
            0,
            RuntimeInfo("test", "second", "cpu", "int8"),
        )

    def test_describe_reports_selected_backend_capabilities(self, tmp_path: Path) -> None:
        """Description resolves the model backend without opening a runtime."""
        coordinator = InferenceCoordinator(tmp_path, drivers=(_Driver(),))

        descriptor = coordinator.describe(_model("local"))

        assert (descriptor.available, descriptor.capabilities.context_biasing) == (
            True,
            True,
        )

    def test_deactivate_releases_active_runtime_idempotently(self, tmp_path: Path) -> None:
        """Deactivation closes the resident session and remains safe when repeated."""
        driver = _Driver()
        coordinator = InferenceCoordinator(tmp_path, drivers=(driver,))
        coordinator.activate(_model("local"))

        coordinator.deactivate()
        coordinator.deactivate()

        assert (driver.sessions[0].close_calls, coordinator.active_runtime) == (
            1,
            None,
        )


def test_fake_session_accepts_the_canonical_request() -> None:
    """The coordinator test session implements the complete transcription protocol."""
    session = _Session(RuntimeInfo("test", "local", "cpu", "int8"))

    result = session.transcribe(TranscriptionRequest(np.zeros(16_000, dtype=np.float32)))

    assert result.duration_s == 1.0


class TestDefaultComposition:
    def test_get_default_coordinator_reuses_matching_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The process composition root is stable while its cache is unchanged."""
        monkeypatch.setattr(composition, "_default_coordinator", None)

        first = get_default_coordinator(tmp_path)
        second = get_default_coordinator(tmp_path)

        assert second is first

    def test_get_default_coordinator_replaces_changed_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Changing the configured cache replaces the process composition root."""
        monkeypatch.setattr(composition, "_default_coordinator", None)

        first = get_default_coordinator(tmp_path / "first")
        second = get_default_coordinator(tmp_path / "second")

        assert (second is not first, second.cache_dir) == (
            True,
            (tmp_path / "second").resolve(),
        )

    def test_activate_model_returns_actual_runtime_info(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The public activation API reports the session's actual execution settings."""
        coordinator = InferenceCoordinator(
            tmp_path,
            drivers=(_Driver("faster-whisper"),),
        )
        monkeypatch.setattr(
            composition,
            "get_default_coordinator",
            lambda: coordinator,
        )

        info = activate_model("tiny", device="cpu", precision="int8")

        assert info == RuntimeInfo("faster-whisper", "tiny", "cpu", "int8")

    def test_deactivate_model_discards_the_process_runtime(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Public deactivation closes the active session and clears the singleton."""
        driver = _Driver("faster-whisper")
        coordinator = InferenceCoordinator(tmp_path, drivers=(driver,))
        coordinator.activate("tiny")
        monkeypatch.setattr(composition, "_default_coordinator", coordinator)

        deactivate_model()

        assert (driver.sessions[0].close_calls, composition._default_coordinator) == (
            1,
            None,
        )

    def test_deactivate_model_discards_runtime_after_close_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A close failure remains visible but cannot leave a stale process runtime."""
        monkeypatch.setattr(
            composition,
            "_default_coordinator",
            _FailingCoordinator(),
        )

        with pytest.raises(RuntimeError, match="close failed"):
            deactivate_model()

        assert composition._default_coordinator is None

    def test_prepare_model_forwards_progress_to_owned_artifact_manager(
        self,
        tmp_path: Path,
    ) -> None:
        """Public preparation forwards the model id and progress callback."""
        with patch.object(composition, "get_default_coordinator") as get_coordinator:
            get_coordinator.return_value.prepare.return_value = tmp_path

            result = prepare_model("tiny", _progress)

        assert (
            result,
            get_coordinator.return_value.prepare.call_args.args,
        ) == (tmp_path, ("tiny", _progress))

    def test_model_is_ready_returns_owned_artifact_status(self) -> None:
        """Public readiness reflects the coordinator's non-downloading check."""
        with patch.object(composition, "get_default_coordinator") as get_coordinator:
            get_coordinator.return_value.is_prepared.return_value = True

            ready = model_is_ready("tiny")

        assert ready is True
