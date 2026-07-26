from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from .contracts import (
    BackendContract,
    BackendDriver,
    RuntimeInfo,
    RuntimeOptions,
    TranscriptionSession,
)
from .errors import BackendLookupError, BackendSessionError, BackendUnavailableError
from ..models import ModelSpec

logger = logging.getLogger(__name__)

type RuntimeIdentity = tuple[ModelSpec, RuntimeOptions]


class DriverRegistry(Protocol):
    """Minimal registry surface required by runtime lifecycle management."""

    def get(self, backend_id: str) -> BackendDriver: ...


class BackendRegistry:
    """Backend drivers indexed by stable identifier."""

    def __init__(self) -> None:
        self._drivers: dict[str, BackendDriver] = {}

    def register(self, driver: BackendDriver) -> None:
        backend_id = driver.id
        if not backend_id.strip():
            raise ValueError("backend id must not be empty")
        if backend_id in self._drivers:
            raise ValueError(f"Backend '{backend_id}' is already registered.")
        self._drivers[backend_id] = driver

    def get(self, backend_id: str) -> BackendDriver:
        try:
            return self._drivers[backend_id]
        except KeyError as exc:
            raise BackendLookupError(f"Backend '{backend_id}' is not registered.") from exc

    def list(self) -> tuple[str, ...]:
        return tuple(self._drivers)


@dataclass(frozen=True, slots=True)
class RuntimeDescriptor:
    """Model and backend capabilities available before a session is opened."""

    model: ModelSpec
    contract: BackendContract
    available: bool


@dataclass(frozen=True, slots=True)
class ActiveRuntime:
    """Read-only description of the currently resident model runtime."""

    model: ModelSpec
    options: RuntimeOptions
    info: RuntimeInfo
    contract: BackendContract


@dataclass(slots=True)
class _ResidentSession:
    identity: RuntimeIdentity
    runtime: ActiveRuntime
    session: TranscriptionSession


class RuntimeManager:
    """Keep at most one model session resident to protect limited VRAM.

    Driver registration is explicit. Backend discovery and availability checks
    happen before inference, never inside a session's transcription hot path.
    """

    def __init__(self, registry: DriverRegistry) -> None:
        self._registry = registry
        self._availability: dict[str, bool] = {}
        self._resident: _ResidentSession | None = None
        self._lock = RLock()

    @property
    def active(self) -> ActiveRuntime | None:
        """Return metadata for the resident runtime, if one is open."""
        with self._lock:
            return self._resident.runtime if self._resident is not None else None

    def describe(self, model: ModelSpec) -> RuntimeDescriptor:
        """Return availability and capability metadata without opening a model."""
        with self._lock:
            driver = self._registry.get(model.backend_id)
            available = self._check_available(driver)
            return RuntimeDescriptor(
                model=model,
                contract=driver.contract,
                available=available,
            )

    def open(
        self,
        model: ModelSpec,
        options: RuntimeOptions | None = None,
    ) -> TranscriptionSession:
        """Reuse the requested runtime or replace the sole resident session."""
        requested = options if options is not None else RuntimeOptions()
        identity = _runtime_identity(model, requested)

        with self._lock:
            resident = self._resident
            if resident is not None and resident.identity == identity:
                logger.debug(
                    "Reusing active inference runtime: backend=%s model=%s device=%s precision=%s",
                    resident.runtime.info.backend_id,
                    resident.runtime.info.model_id,
                    resident.runtime.info.device,
                    resident.runtime.info.precision,
                )
                return resident.session

            driver = self._registry.get(model.backend_id)
            self._require_available(driver)
            if resident is not None:
                logger.info(
                    "Switching inference runtime: backend=%s model=%s -> backend=%s model=%s",
                    resident.runtime.info.backend_id,
                    resident.runtime.info.model_id,
                    model.backend_id,
                    model.id,
                )
            self._close_active()

            prepared = driver.prepare(model)
            try:
                session = driver.open(prepared, requested)
            except Exception as exc:
                raise BackendSessionError(f"Backend '{model.backend_id}' could not open model '{model.id}'.") from exc

            try:
                info = _validate_identity(session, model)
            except Exception:
                _close_invalid_session(session, model)
                raise

            runtime = ActiveRuntime(
                model=model,
                options=requested,
                info=info,
                contract=driver.contract,
            )
            self._resident = _ResidentSession(
                identity=identity,
                runtime=runtime,
                session=session,
            )
            logger.info(
                "Activated inference runtime: backend=%s model=%s device=%s precision=%s fallback_to_cpu=%s",
                info.backend_id,
                info.model_id,
                info.device,
                info.precision,
                info.fallback_to_cpu,
            )
            return session

    def close_model(self, model_id: str) -> None:
        """Close the active runtime when it belongs to the requested model."""
        with self._lock:
            resident = self._resident
            if resident is not None and resident.runtime.model.id == model_id:
                self._close_active()

    def close_all(self) -> None:
        """Close the active runtime and clear all lifecycle state."""
        with self._lock:
            self._close_active()

    def _check_available(self, driver: BackendDriver) -> bool:
        backend_id = driver.id
        if backend_id not in self._availability:
            try:
                self._availability[backend_id] = driver.is_available()
            except Exception as exc:
                self._availability[backend_id] = False
                raise BackendUnavailableError(f"Backend '{backend_id}' failed its availability check.") from exc
        return self._availability[backend_id]

    def _require_available(self, driver: BackendDriver) -> None:
        if not self._check_available(driver):
            raise BackendUnavailableError(f"Backend '{driver.id}' is unavailable.")

    def _close_active(self) -> None:
        resident = self._resident
        if resident is None:
            return

        self._resident = None
        try:
            resident.session.close()
        except Exception as exc:
            model = resident.runtime.model
            raise BackendSessionError(
                f"Failed to close active backend '{model.backend_id}' model '{model.id}'."
            ) from exc
        logger.info(
            "Unloaded inference runtime: backend=%s model=%s device=%s precision=%s",
            resident.runtime.info.backend_id,
            resident.runtime.info.model_id,
            resident.runtime.info.device,
            resident.runtime.info.precision,
        )


def _runtime_identity(model: ModelSpec, options: RuntimeOptions) -> RuntimeIdentity:
    return model, options


def _validate_identity(session: TranscriptionSession, model: ModelSpec) -> RuntimeInfo:
    try:
        info = session.info
    except Exception as exc:
        raise BackendSessionError(
            f"Backend '{model.backend_id}' did not report runtime information for model '{model.id}'."
        ) from exc

    if info.backend_id != model.backend_id:
        raise BackendSessionError(
            f"Backend session identity mismatch: expected '{model.backend_id}', got '{info.backend_id}'."
        )
    if info.model_id != model.id:
        raise BackendSessionError(f"Model session identity mismatch: expected '{model.id}', got '{info.model_id}'.")
    return info


def _close_invalid_session(session: TranscriptionSession, model: ModelSpec) -> None:
    try:
        session.close()
    except Exception as exc:
        logger.warning(
            "Failed to close invalid backend session: backend=%s model=%s error=%s",
            model.backend_id,
            model.id,
            exc,
        )


__all__ = [
    "ActiveRuntime",
    "BackendRegistry",
    "RuntimeDescriptor",
    "RuntimeManager",
]
