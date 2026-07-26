from pathlib import Path
from typing import cast

import pytest
from voicepad_core.audio import WaveformSpec
from voicepad_core.inference.contracts import (
    BackendContract,
    PreparedModel,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionRequest,
)
from voicepad_core.inference.errors import BackendLookupError, BackendSessionError, BackendUnavailableError
from voicepad_core.inference.runtime import RuntimeManager
from voicepad_core.inference.types import TranscriptionResult
from voicepad_core.models import Model


class _Session:
    def __init__(self, model_id: str, backend: str = "test") -> None:
        self._info = RuntimeInfo(backend, model_id, "cuda", "float16")
        self.closed = False

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


class _Driver:
    id = "test"
    contract = BackendContract(WaveformSpec(16_000))

    def __init__(self, *, available: bool = True, reported_backend: str = "test") -> None:
        self.available = available
        self.reported_backend = reported_backend
        self.sessions: list[_Session] = []

    def is_available(self) -> bool:
        return self.available

    def open(self, model: PreparedModel, options: RuntimeOptions) -> _Session:
        session = _Session(model.spec.id, self.reported_backend)
        self.sessions.append(session)
        return session


def _model(model_id: str = "tiny", backend: str = "test") -> Model:
    return Model(model_id, f"owner/{model_id}", backend, ("model.bin",), model_id, "test")


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Avoid network access while testing runtime lifecycle behavior."""
    path = tmp_path / "model"
    path.mkdir()
    (path / "model.bin").write_bytes(b"model")
    monkeypatch.setattr("voicepad_core.inference.runtime.prepare_artifact", lambda *_args, **_kwargs: path)
    return path


def test_contract_comes_from_selected_backend(tmp_path: Path) -> None:
    """The model backend selects the published waveform contract."""
    manager = RuntimeManager(tmp_path, [_Driver()])
    assert manager.contract(_model()).audio.sample_rate == 16_000


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    """A model cannot run without its explicitly selected backend."""
    with pytest.raises(BackendLookupError):
        RuntimeManager(tmp_path, []).contract(_model(backend="missing"))


def test_open_reuses_identical_runtime(tmp_path: Path, prepared: Path) -> None:
    """Repeated requests for the same model and options reuse GPU state."""
    driver = _Driver()
    manager = RuntimeManager(tmp_path, [driver])
    first = manager.open(_model())
    second = manager.open(_model())
    assert first is second


def test_open_replaces_previous_runtime(tmp_path: Path, prepared: Path) -> None:
    """Switching models closes the previous resident session."""
    driver = _Driver()
    manager = RuntimeManager(tmp_path, [driver])
    first = cast(_Session, manager.open(_model("first")))
    manager.open(_model("second"))
    assert first.closed is True


def test_unavailable_backend_fails_before_download(tmp_path: Path) -> None:
    """Unavailable native dependencies produce an explicit error."""
    manager = RuntimeManager(tmp_path, [_Driver(available=False)])
    with pytest.raises(BackendUnavailableError):
        manager.open(_model())


def test_wrong_session_identity_is_rejected(tmp_path: Path, prepared: Path) -> None:
    """A backend cannot silently report a different runtime."""
    manager = RuntimeManager(tmp_path, [_Driver(reported_backend="wrong")])
    with pytest.raises(BackendSessionError, match="wrong runtime identity"):
        manager.open(_model())


def test_close_is_idempotent(tmp_path: Path, prepared: Path) -> None:
    """Closing repeatedly leaves no resident model."""
    manager = RuntimeManager(tmp_path, [_Driver()])
    session = cast(_Session, manager.open(_model()))
    manager.close()
    manager.close()
    assert (session.closed, manager.active_info) == (True, None)
