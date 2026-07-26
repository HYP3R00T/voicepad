from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from voicepad_core.audio import WaveformSpec
from voicepad_core.inference.contracts import (
    BackendCapabilities,
    BackendContract,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from voicepad_core.inference.errors import (
    BackendLookupError,
    BackendSessionError,
    BackendUnavailableError,
)
from voicepad_core.inference.runtime import BackendRegistry, RuntimeManager
from voicepad_core.inference.types import TranscriptionResult
from voicepad_core.models import HuggingFaceArtifact, ModelSpec


class FakeSession:
    def __init__(self, info: RuntimeInfo, *, close_error: Exception | None = None) -> None:
        self._info = info
        self._close_error = close_error
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
        if self._close_error is not None:
            raise self._close_error


class FakeDriver:
    def __init__(
        self,
        backend_id: str,
        *,
        available: bool = True,
        actual_device: str | None = None,
        actual_precision: str | None = None,
        reported_backend_id: str | None = None,
        reported_model_id: str | None = None,
    ) -> None:
        self._id = backend_id
        self._available = available
        self._actual_device = actual_device
        self._actual_precision = actual_precision
        self._reported_backend_id = reported_backend_id
        self._reported_model_id = reported_model_id
        self.availability_checks = 0
        self.prepared: list[ModelSpec] = []
        self.opened: list[tuple[PreparedModel, RuntimeOptions]] = []
        self.sessions: list[FakeSession] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    @property
    def contract(self) -> BackendContract:
        return BackendContract(WaveformSpec(16_000), self.capabilities)

    def is_available(self) -> bool:
        self.availability_checks += 1
        return self._available

    def prepare(self, model: ModelSpec) -> PreparedModel:
        self.prepared.append(model)
        return PreparedModel(model, Path(f"{model.id}.bin"))

    def open(self, model: PreparedModel, options: RuntimeOptions) -> FakeSession:
        self.opened.append((model, options))
        info = RuntimeInfo(
            backend_id=self._reported_backend_id or self.id,
            model_id=self._reported_model_id or model.spec.id,
            device=self._actual_device or options.device,
            precision=self._actual_precision or options.precision,
            fallback_to_cpu=self._actual_device == "cpu" and options.device != "cpu",
        )
        session = FakeSession(info)
        self.sessions.append(session)
        return session


def model_spec(model_id: str = "tiny", backend_id: str = "test") -> ModelSpec:
    return ModelSpec(
        model_id,
        HuggingFaceArtifact(f"owner/{model_id}"),
        backend_id=backend_id,
    )


class TestBackendRegistry:
    def test_register_get_and_list_are_explicit(self) -> None:
        """Registered drivers are retrieved and listed in insertion order."""
        registry = BackendRegistry()
        first = FakeDriver("first")
        second = FakeDriver("second")

        registry.register(first)
        registry.register(second)

        assert registry.get("first") is first
        assert registry.list() == ("first", "second")

    def test_duplicate_backend_is_rejected(self) -> None:
        """Registering the same backend identifier twice raises an error."""
        registry = BackendRegistry()
        registry.register(FakeDriver("test"))

        with pytest.raises(ValueError, match="already registered"):
            registry.register(FakeDriver("test"))

    def test_blank_backend_id_is_rejected(self) -> None:
        """A driver with a blank identifier cannot be registered."""
        registry = BackendRegistry()

        with pytest.raises(ValueError, match="backend id"):
            registry.register(FakeDriver(" "))

    def test_missing_backend_raises_lookup_error(self) -> None:
        """Looking up an unregistered backend raises the domain lookup error."""
        registry = BackendRegistry()

        with pytest.raises(BackendLookupError, match="not registered"):
            registry.get("missing")


class TestRuntimeManager:
    def test_resolves_driver_from_model_backend_id(self) -> None:
        """Opening a model selects its explicitly registered backend driver."""
        registry = BackendRegistry()
        driver = FakeDriver("selected")
        registry.register(driver)
        manager = RuntimeManager(registry)
        model = model_spec(backend_id="selected")

        session = manager.open(model, RuntimeOptions(device="cuda", precision="int8"))

        assert session is driver.sessions[0]
        assert driver.prepared == [model]

    def test_reuses_same_identity_and_closes_before_switching(self) -> None:
        """The same runtime is reused while changed options replace and close it."""
        registry = BackendRegistry()
        driver = FakeDriver("test")
        registry.register(driver)
        manager = RuntimeManager(registry)
        model = model_spec()

        first = manager.open(model, RuntimeOptions(device="cuda", precision="int8"))
        repeated = manager.open(model, RuntimeOptions(device="cuda", precision="int8"))
        other_device = manager.open(model, RuntimeOptions(device="cpu", precision="int8"))

        assert repeated is first
        assert other_device is not first
        assert cast(FakeSession, first).close_calls == 1
        assert len(driver.opened) == 2

    def test_availability_is_checked_once_per_backend(self) -> None:
        """Backend availability is cached across model switches."""
        registry = BackendRegistry()
        driver = FakeDriver("test")
        registry.register(driver)
        manager = RuntimeManager(registry)

        manager.open(model_spec("tiny"))
        manager.open(model_spec("base"))

        assert driver.availability_checks == 1

    def test_unavailable_backend_is_not_prepared_or_opened(self) -> None:
        """An unavailable backend fails before preparing or opening a model."""
        registry = BackendRegistry()
        driver = FakeDriver("test", available=False)
        registry.register(driver)
        manager = RuntimeManager(registry)

        with pytest.raises(BackendUnavailableError, match="unavailable"):
            manager.open(model_spec())
        with pytest.raises(BackendUnavailableError, match="unavailable"):
            manager.open(model_spec("base"))

        assert driver.availability_checks == 1
        assert driver.prepared == []
        assert driver.opened == []

    def test_actual_runtime_device_and_precision_are_authoritative(self) -> None:
        """Session-reported execution details override requested runtime options."""
        registry = BackendRegistry()
        driver = FakeDriver("test", actual_device="cpu", actual_precision="int8")
        registry.register(driver)
        manager = RuntimeManager(registry)

        session = manager.open(
            model_spec(),
            RuntimeOptions(device="cuda", precision="float16", allow_cpu_fallback=True),
        )

        assert session.info.device == "cpu"
        assert session.info.precision == "int8"
        assert session.info.fallback_to_cpu is True

    @pytest.mark.parametrize(
        ("reported_backend_id", "reported_model_id", "message"),
        [
            ("wrong", None, "Backend session identity mismatch"),
            (None, "wrong", "Model session identity mismatch"),
        ],
    )
    def test_rejects_and_closes_mismatched_session_identity(
        self,
        reported_backend_id: str | None,
        reported_model_id: str | None,
        message: str,
    ) -> None:
        """A session reporting the wrong backend or model is rejected and closed."""
        registry = BackendRegistry()
        driver = FakeDriver(
            "test",
            reported_backend_id=reported_backend_id,
            reported_model_id=reported_model_id,
        )
        registry.register(driver)
        manager = RuntimeManager(registry)

        with pytest.raises(BackendSessionError, match=message):
            manager.open(model_spec())

        assert driver.sessions[0].close_calls == 1

    def test_open_failure_raises_session_error(self) -> None:
        """A driver open failure is wrapped in a backend session error."""

        class FailingDriver(FakeDriver):
            def open(self, model: PreparedModel, options: RuntimeOptions) -> FakeSession:
                raise RuntimeError("runtime failed")

        registry = BackendRegistry()
        registry.register(FailingDriver("test"))
        manager = RuntimeManager(registry)

        with pytest.raises(BackendSessionError, match="could not open") as exc_info:
            manager.open(model_spec())

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_close_model_only_closes_matching_active_model(self) -> None:
        """Closing a model leaves a different active model resident."""
        registry = BackendRegistry()
        driver = FakeDriver("test")
        registry.register(driver)
        manager = RuntimeManager(registry)
        active = cast(FakeSession, manager.open(model_spec("base")))

        manager.close_model("tiny")

        assert active.close_calls == 0
        assert manager.open(model_spec("base")) is active

        manager.close_model("base")

        assert active.close_calls == 1

    def test_close_all_failure_still_clears_active_runtime(self) -> None:
        """A close failure is reported after the active runtime is cleared."""
        registry = BackendRegistry()
        driver = FakeDriver("test")
        registry.register(driver)
        manager = RuntimeManager(registry)
        first = cast(FakeSession, manager.open(model_spec("tiny")))
        first._close_error = RuntimeError("close failed")

        with pytest.raises(BackendSessionError, match="Failed to close active"):
            manager.close_all()

        assert first.close_calls == 1
        assert manager.open(model_spec("tiny")) is not first

    def test_close_operations_are_safe_when_no_sessions_are_open(self) -> None:
        """Close operations are idempotent when no runtime is active."""
        manager = RuntimeManager(BackendRegistry())

        manager.close_model("missing")
        manager.close_all()


def test_fake_session_transcribes_canonical_request() -> None:
    """The fake session used by lifecycle tests accepts the canonical request."""
    request = TranscriptionRequest(np.zeros(16_000, dtype=np.float32), sample_rate=16_000)
    session = FakeSession(RuntimeInfo("test", "tiny", "cpu", "int8"))

    result = session.transcribe(request)

    assert result.duration_s == 1.0
