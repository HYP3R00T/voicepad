from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import cast

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
from voicepad_core.inference.errors import BackendSessionError
from voicepad_core.inference.runtime import BackendRegistry, RuntimeManager
from voicepad_core.inference.types import TranscriptionResult
from voicepad_core.models import HuggingFaceArtifact, ModelSpec


class _Session:
    def __init__(self, info: RuntimeInfo) -> None:
        self._info = info
        self.close_calls = 0

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

    def close(self) -> None:
        self.close_calls += 1


class _Driver:
    def __init__(self) -> None:
        self.availability_checks = 0
        self.open_calls = 0
        self.sessions: list[_Session] = []

    @property
    def id(self) -> str:
        return "test"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            word_timestamps=True,
            language_detection=True,
            context_biasing=True,
        )

    @property
    def contract(self) -> BackendContract:
        return BackendContract(WaveformSpec(16_000), self.capabilities)

    def is_available(self) -> bool:
        self.availability_checks += 1
        return True

    def prepare(self, model: ModelSpec) -> PreparedModel:
        return PreparedModel(model, Path(f"{model.id}.bin"))

    def open(self, model: PreparedModel, options: RuntimeOptions) -> _Session:
        self.open_calls += 1
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


def _model(model_id: str = "tiny") -> ModelSpec:
    return ModelSpec(
        model_id,
        HuggingFaceArtifact(f"owner/{model_id}"),
        backend_id="test",
    )


def _manager() -> tuple[RuntimeManager, _Driver]:
    registry = BackendRegistry()
    driver = _Driver()
    registry.register(driver)
    return RuntimeManager(registry), driver


def test_describe_exposes_availability_and_capabilities_without_opening() -> None:
    """Describing a model reports cached backend metadata without opening it."""
    manager, driver = _manager()

    descriptor = manager.describe(_model())

    assert descriptor.available is True
    assert descriptor.model.id == "tiny"
    assert descriptor.contract.decoding.word_timestamps is True
    assert descriptor.contract.decoding.context_biasing is True
    assert driver.availability_checks == 1
    assert driver.open_calls == 0


def test_active_runtime_reports_model_options_info_and_capabilities() -> None:
    """An open runtime exposes its model, options, resolved info, and capabilities."""
    manager, _ = _manager()
    options = RuntimeOptions(device="cuda", precision="int8_float16")

    manager.open(_model(), options)

    assert manager.active is not None
    assert manager.active.model.id == "tiny"
    assert manager.active.options == options
    assert manager.active.info.device == "cuda"
    assert manager.active.contract.decoding.language_detection is True


def test_switch_closes_old_session_before_driver_opens_new_session() -> None:
    """Switching models closes the resident session before opening the replacement."""
    events: list[str] = []

    class OrderedSession(_Session):
        def close(self) -> None:
            events.append(f"close:{self.info.model_id}")
            super().close()

    class OrderedDriver(_Driver):
        def open(self, model: PreparedModel, options: RuntimeOptions) -> _Session:
            events.append(f"open:{model.spec.id}")
            session = OrderedSession(RuntimeInfo(self.id, model.spec.id, options.device, options.precision))
            self.sessions.append(session)
            return session

    registry = BackendRegistry()
    driver = OrderedDriver()
    registry.register(driver)
    manager = RuntimeManager(registry)

    manager.open(_model("tiny"))
    manager.open(_model("base"))

    assert events == ["open:tiny", "close:tiny", "open:base"]


def test_concurrent_open_of_same_identity_creates_one_session() -> None:
    """Concurrent opens for one identity share exactly one resident session."""
    manager, driver = _manager()
    model = _model()

    with ThreadPoolExecutor(max_workers=8) as executor:
        sessions = list(executor.map(lambda _: manager.open(model), range(32)))

    assert all(session is sessions[0] for session in sessions)
    assert driver.open_calls == 1


def test_close_all_removes_active_runtime() -> None:
    """Closing all unloads the sole resident session and clears active metadata."""
    manager, driver = _manager()
    manager.open(_model())

    manager.close_all()

    assert manager.active is None
    assert driver.sessions[0].close_calls == 1


def test_open_failure_after_switch_leaves_no_runtime_resident() -> None:
    """When replacement loading fails, the old runtime is closed and none remains."""

    class FailingReplacementDriver(_Driver):
        def open(self, model: PreparedModel, options: RuntimeOptions) -> _Session:
            if model.spec.id == "base":
                raise RuntimeError("load failed")
            return super().open(model, options)

    registry = BackendRegistry()
    driver = FailingReplacementDriver()
    registry.register(driver)
    manager = RuntimeManager(registry)
    old_session = cast(_Session, manager.open(_model("tiny")))

    with pytest.raises(BackendSessionError, match="could not open"):
        manager.open(_model("base"))

    assert old_session.close_calls == 1
    assert manager.active is None


def test_close_failure_prevents_replacement_from_opening() -> None:
    """When unloading fails during a switch, no replacement runtime is opened."""

    class FailingCloseSession(_Session):
        def close(self) -> None:
            super().close()
            raise RuntimeError("unload failed")

    class FailingCloseDriver(_Driver):
        def open(self, model: PreparedModel, options: RuntimeOptions) -> _Session:
            self.open_calls += 1
            session = FailingCloseSession(RuntimeInfo(self.id, model.spec.id, options.device, options.precision))
            self.sessions.append(session)
            return session

    registry = BackendRegistry()
    driver = FailingCloseDriver()
    registry.register(driver)
    manager = RuntimeManager(registry)
    manager.open(_model("tiny"))

    with pytest.raises(BackendSessionError, match="Failed to close active"):
        manager.open(_model("base"))

    assert driver.open_calls == 1
    assert manager.active is None


def test_close_waits_for_in_progress_open_then_unloads_once() -> None:
    """A concurrent close waits for model opening and then unloads it exactly once."""

    class BlockingDriver(_Driver):
        def __init__(self) -> None:
            super().__init__()
            self.open_started = Event()
            self.release_open = Event()

        def open(self, model: PreparedModel, options: RuntimeOptions) -> _Session:
            session = super().open(model, options)
            self.open_started.set()
            if not self.release_open.wait(timeout=2):
                raise RuntimeError("test timed out waiting to release open")
            return session

    registry = BackendRegistry()
    driver = BlockingDriver()
    registry.register(driver)
    manager = RuntimeManager(registry)

    with ThreadPoolExecutor(max_workers=2) as executor:
        opening = executor.submit(manager.open, _model())
        assert driver.open_started.wait(timeout=2)
        closing = executor.submit(manager.close_all)
        assert not closing.done()
        driver.release_open.set()
        session = cast(_Session, opening.result(timeout=2))
        closing.result(timeout=2)

    assert session.close_calls == 1
    assert manager.active is None


def test_lifecycle_logs_activation_switch_and_unload_once(caplog: pytest.LogCaptureFixture) -> None:
    """Normal model switching emits one activation, switch, and unload event per action."""
    manager, _ = _manager()

    with caplog.at_level(logging.INFO, logger="voicepad_core.inference.runtime"):
        manager.open(_model("tiny"))
        manager.open(_model("base"))
        manager.close_all()

    messages = [record.getMessage() for record in caplog.records]
    assert sum(message.startswith("Activated inference runtime") for message in messages) == 2
    assert sum(message.startswith("Switching inference runtime") for message in messages) == 1
    assert sum(message.startswith("Unloaded inference runtime") for message in messages) == 2
