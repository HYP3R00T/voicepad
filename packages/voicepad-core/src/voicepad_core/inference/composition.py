from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from threading import RLock

from .artifacts import ProgressCallback, locate_artifact, prepare_artifact
from .backends import FasterWhisperDriver, ParakeetOnnxDriver
from .contracts import (
    BackendDriver,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionSession,
)
from .runtime import ActiveRuntime, BackendRegistry, RuntimeDescriptor, RuntimeManager
from ..config import get_config
from ..models import ModelSpec, resolve_model_spec

logger = logging.getLogger(__name__)


class InferenceCoordinator:
    """Own model artifacts, backend selection, and the sole resident runtime."""

    def __init__(
        self,
        cache_dir: Path,
        drivers: Iterable[BackendDriver] | None = None,
    ) -> None:
        self._cache_dir = cache_dir.expanduser().resolve()
        registry = BackendRegistry()
        enabled_drivers = (
            tuple(drivers)
            if drivers is not None
            else (
                FasterWhisperDriver(self._cache_dir),
                ParakeetOnnxDriver(self._cache_dir),
            )
        )
        for driver in enabled_drivers:
            registry.register(driver)
        self._registry = registry
        self._sessions = RuntimeManager(registry)

    @property
    def cache_dir(self) -> Path:
        """Return the model artifact cache owned by this coordinator."""
        return self._cache_dir

    @property
    def backend_ids(self) -> tuple[str, ...]:
        """Return the explicitly enabled backend identifiers."""
        return self._registry.list()

    @property
    def active_runtime(self) -> ActiveRuntime | None:
        """Return the currently resident runtime, if one is active."""
        return self._sessions.active

    @property
    def active_info(self) -> RuntimeInfo | None:
        """Return actual execution information for the resident runtime."""
        active = self.active_runtime
        return active.info if active is not None else None

    def describe(self, model: str | ModelSpec) -> RuntimeDescriptor:
        """Resolve a model and report its backend capabilities and availability."""
        return self._sessions.describe(self._resolve(model))

    def is_prepared(self, model: str | ModelSpec) -> bool:
        """Return whether a complete local artifact is ready without downloading."""
        return locate_artifact(self._resolve(model), self._cache_dir) is not None

    def prepare(
        self,
        model: str | ModelSpec,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Acquire and validate a model artifact without opening its runtime."""
        resolved = self._resolve(model)
        artifact_path = prepare_artifact(resolved, self._cache_dir, on_progress)
        logger.info(
            "Prepared inference artifact: backend=%s model=%s path=%s",
            resolved.backend_id,
            resolved.id,
            artifact_path,
        )
        return artifact_path

    def activate(
        self,
        model: str | ModelSpec,
        options: RuntimeOptions | None = None,
    ) -> TranscriptionSession:
        """Open or reuse the requested model as the sole resident runtime."""
        return self._sessions.open(self._resolve(model), options)

    def deactivate(self) -> None:
        """Close the resident runtime and release its resources."""
        self._sessions.close_all()

    @staticmethod
    def _resolve(model: str | ModelSpec) -> ModelSpec:
        return resolve_model_spec(model) if isinstance(model, str) else model


_default_coordinator: InferenceCoordinator | None = None
_default_coordinator_lock = RLock()


def get_default_coordinator(cache_dir: Path | None = None) -> InferenceCoordinator:
    """Return the process coordinator, replacing it when its cache changes."""
    global _default_coordinator
    resolved_cache_dir = (cache_dir or get_config().model_cache_path).expanduser().resolve()
    with _default_coordinator_lock:
        if _default_coordinator is None or _default_coordinator.cache_dir != resolved_cache_dir:
            if _default_coordinator is not None:
                _default_coordinator.deactivate()
            _default_coordinator = InferenceCoordinator(resolved_cache_dir)
        return _default_coordinator


def prepare_model(
    model_name: str,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Acquire and validate a catalogue model without opening its runtime."""
    return get_default_coordinator().prepare(model_name, on_progress)


def model_is_ready(model_name: str) -> bool:
    """Return whether a complete catalogue model artifact is locally available."""
    return get_default_coordinator().is_prepared(model_name)


def activate_model(
    model_name: str,
    device: str = "auto",
    precision: str = "auto",
) -> RuntimeInfo:
    """Activate one catalogue model and return its actual runtime settings."""
    session = get_default_coordinator().activate(
        model_name,
        RuntimeOptions(device=device, precision=precision),
    )
    return session.info


def deactivate_model() -> None:
    """Close and discard the process-wide inference runtime."""
    global _default_coordinator
    with _default_coordinator_lock:
        if _default_coordinator is None:
            return
        coordinator = _default_coordinator
        _default_coordinator = None
        coordinator.deactivate()


__all__ = [
    "InferenceCoordinator",
    "activate_model",
    "deactivate_model",
    "get_default_coordinator",
    "model_is_ready",
    "prepare_model",
]
