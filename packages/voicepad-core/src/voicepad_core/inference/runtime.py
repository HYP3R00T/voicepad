from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from threading import RLock

from .artifacts import ProgressCallback, locate_artifact, prepare_artifact
from .backends import FasterWhisperDriver, ParakeetOnnxDriver
from .contracts import BackendContract, BackendDriver, PreparedModel, RuntimeInfo, RuntimeOptions, TranscriptionSession
from .errors import BackendLookupError, BackendSessionError, BackendUnavailableError
from ..config import get_config
from ..models import Model, get_model

logger = logging.getLogger(__name__)


class RuntimeManager:
    """Download models and keep one backend session resident in memory."""

    def __init__(
        self,
        cache_dir: Path,
        drivers: Iterable[BackendDriver] | None = None,
    ) -> None:
        self.cache_dir = cache_dir.expanduser().resolve()
        enabled = drivers or (FasterWhisperDriver(), ParakeetOnnxDriver())
        self._drivers = {driver.id: driver for driver in enabled}
        self._identity: tuple[Model, RuntimeOptions] | None = None
        self._session: TranscriptionSession | None = None
        self._lock = RLock()

    @property
    def active_info(self) -> RuntimeInfo | None:
        """Return information about the loaded model."""
        with self._lock:
            return self._session.info if self._session is not None else None

    def contract(self, model: str | Model) -> BackendContract:
        """Return the selected backend's input and output contract."""
        return self._driver(self._model(model)).contract

    def is_ready(self, model: str | Model) -> bool:
        """Return whether a complete model is already cached."""
        return locate_artifact(self._model(model), self.cache_dir) is not None

    def prepare(
        self,
        model: str | Model,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Download and validate a model without loading it."""
        selected = self._model(model)
        path = prepare_artifact(selected, self.cache_dir, on_progress)
        logger.info("Prepared model: backend=%s model=%s path=%s", selected.backend, selected.id, path)
        return path

    def open(
        self,
        model: str | Model,
        options: RuntimeOptions | None = None,
    ) -> TranscriptionSession:
        """Load a model, replacing the current session only when necessary."""
        selected = self._model(model)
        requested = options or RuntimeOptions()
        identity = (selected, requested)

        with self._lock:
            if self._session is not None and self._identity == identity:
                return self._session

            driver = self._driver(selected)
            if not driver.is_available():
                raise BackendUnavailableError(f"Backend '{driver.id}' is unavailable.")

            self.close()
            prepared = PreparedModel(selected, self.prepare(selected))
            try:
                session = driver.open(prepared, requested)
            except Exception as exc:
                raise BackendSessionError(f"Backend '{driver.id}' could not open model '{selected.id}'.") from exc

            info = session.info
            if info.backend_id != selected.backend or info.model_id != selected.id:
                session.close()
                raise BackendSessionError(f"Backend '{driver.id}' returned the wrong runtime identity.")

            self._identity = identity
            self._session = session
            logger.info(
                "Activated runtime: backend=%s model=%s device=%s precision=%s",
                info.backend_id,
                info.model_id,
                info.device,
                info.precision,
            )
            return session

    def close(self) -> None:
        """Release the loaded model."""
        with self._lock:
            session = self._session
            self._session = None
            self._identity = None
            if session is None:
                return
            try:
                session.close()
            except Exception as exc:
                raise BackendSessionError("Failed to close the active backend session.") from exc

    def _driver(self, model: Model) -> BackendDriver:
        try:
            return self._drivers[model.backend]
        except KeyError as exc:
            raise BackendLookupError(f"Backend '{model.backend}' is not registered.") from exc

    @staticmethod
    def _model(model: str | Model) -> Model:
        return get_model(model) if isinstance(model, str) else model


_default_manager: RuntimeManager | None = None
_default_lock = RLock()


def get_runtime_manager(cache_dir: Path | None = None) -> RuntimeManager:
    """Return the process-wide runtime manager for the selected cache."""
    global _default_manager
    selected_cache = (cache_dir or get_config().model_cache_path).expanduser().resolve()
    with _default_lock:
        if _default_manager is None or _default_manager.cache_dir != selected_cache:
            if _default_manager is not None:
                _default_manager.close()
            _default_manager = RuntimeManager(selected_cache)
        return _default_manager


def prepare_model(model_name: str, on_progress: ProgressCallback | None = None) -> Path:
    """Download a supported model."""
    return get_runtime_manager().prepare(model_name, on_progress)


def model_is_ready(model_name: str) -> bool:
    """Return whether a supported model is cached."""
    return get_runtime_manager().is_ready(model_name)


def activate_model(model_name: str, device: str = "auto", precision: str = "auto") -> RuntimeInfo:
    """Load a supported model and report the selected runtime."""
    return get_runtime_manager().open(model_name, RuntimeOptions(device=device, precision=precision)).info


def deactivate_model() -> None:
    """Close and discard the process-wide runtime."""
    global _default_manager
    with _default_lock:
        manager = _default_manager
        _default_manager = None
        if manager is not None:
            manager.close()


__all__ = [
    "RuntimeManager",
    "activate_model",
    "deactivate_model",
    "get_runtime_manager",
    "model_is_ready",
    "prepare_model",
]
