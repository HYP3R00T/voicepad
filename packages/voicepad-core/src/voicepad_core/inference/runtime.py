from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from threading import Lock, RLock
from typing import Protocol

from voicepad_core.artifacts import ArtifactStore, ProgressCallback
from voicepad_core.deployments import ArtifactManifest, DeploymentDefinition, get_deployment, get_manifest
from voicepad_core.preprocessing import PreprocessedAudio

from .cuda import CudaDevice, admit_cuda_device
from .parakeet import TransformersParakeetTDTSession
from .types import (
    ActiveDeployment,
    BackendResult,
    CancellationToken,
    InferenceError,
    InvalidTranscriptionInputError,
    RuntimeBusyError,
    TranscriptionIntent,
    TranscriptionSession,
    UnsupportedIntentError,
)

SessionFactory = Callable[[DeploymentDefinition, ArtifactManifest, Path, CudaDevice], TranscriptionSession]
DeviceAdmitter = Callable[[DeploymentDefinition, int], CudaDevice]


class ArtifactProvider(Protocol):
    def prepare(
        self,
        manifest: ArtifactManifest,
        on_progress: ProgressCallback | None = None,
    ) -> Path: ...


class EngineState(StrEnum):
    UNPREPARED = "unprepared"
    PREPARING_ARTIFACT = "preparing-artifact"
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    ACTIVE_JOB = "active-job"
    UNLOADING = "unloading"
    FAILED = "failed"


class ResidentTranscriptionEngine:
    """Own one warmed deployment session and serialize all model operations."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        session_factories: Mapping[str, SessionFactory] | None = None,
        device_admitter: DeviceAdmitter = admit_cuda_device,
        artifact_store: ArtifactProvider | None = None,
    ) -> None:
        self._store = artifact_store or ArtifactStore(artifact_root)
        self._factories = dict(
            session_factories
            or {
                "transformers-parakeet-tdt": TransformersParakeetTDTSession,
            }
        )
        self._admit_device = device_admitter
        self._state = EngineState.UNPREPARED
        self._session: TranscriptionSession | None = None
        self._state_lock = RLock()
        self._operation_lock = Lock()

    @property
    def state(self) -> EngineState:
        with self._state_lock:
            return self._state

    @property
    def active_deployment(self) -> ActiveDeployment | None:
        with self._state_lock:
            return self._session.deployment if self._session is not None else None

    def activate(
        self,
        deployment_id: str,
        *,
        device_index: int = 0,
        on_progress: ProgressCallback | None = None,
    ) -> ActiveDeployment:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeBusyError("The transcription engine is busy.")
        candidate: TranscriptionSession | None = None
        try:
            definition = get_deployment(deployment_id)
            manifest = get_manifest(definition.artifact_manifest_id)
            with self._state_lock:
                if (
                    self._session is not None
                    and self._session.deployment.definition.id == deployment_id
                    and self._state is EngineState.READY
                ):
                    return self._session.deployment
                self._close_session_locked()
                self._state = EngineState.PREPARING_ARTIFACT

            snapshot = self._store.prepare(manifest, on_progress)
            device = self._admit_device(definition, device_index)
            factory = self._factories.get(definition.adapter_id)
            if factory is None:
                raise InferenceError(f"No session adapter is registered for '{definition.adapter_id}'.")
            with self._state_lock:
                self._state = EngineState.LOADING
            candidate = factory(definition, manifest, snapshot, device)
            with self._state_lock:
                self._state = EngineState.WARMING
            warm = getattr(candidate, "warm", None)
            if not callable(warm):
                raise InferenceError(f"Session adapter '{definition.adapter_id}' does not implement warm-up.")
            warm()
            with self._state_lock:
                self._session = candidate
                candidate = None
                self._state = EngineState.READY
                return self._session.deployment
        except Exception:
            if candidate is not None:
                candidate.close()
            with self._state_lock:
                self._session = None
                self._state = EngineState.FAILED
            raise
        finally:
            self._operation_lock.release()

    def transcribe(
        self,
        audio: PreprocessedAudio,
        intent: TranscriptionIntent | None = None,
        cancellation: CancellationToken | None = None,
    ) -> BackendResult:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeBusyError("The transcription engine is busy.")
        try:
            with self._state_lock:
                if self._state is not EngineState.READY or self._session is None:
                    raise InferenceError(f"The transcription engine is not ready: state={self._state}.")
                session = self._session
                self._state = EngineState.ACTIVE_JOB
            try:
                result = session.transcribe(
                    audio,
                    intent or TranscriptionIntent(),
                    cancellation or CancellationToken(),
                )
            except (InvalidTranscriptionInputError, UnsupportedIntentError):
                with self._state_lock:
                    self._state = EngineState.READY
                raise
            except Exception:
                with self._state_lock:
                    self._close_session_locked()
                    self._state = EngineState.FAILED
                raise
            with self._state_lock:
                self._state = EngineState.READY
            return result
        finally:
            self._operation_lock.release()

    def unload(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeBusyError("The transcription engine is busy.")
        try:
            with self._state_lock:
                self._state = EngineState.UNLOADING
                self._close_session_locked()
                self._state = EngineState.UNPREPARED
        finally:
            self._operation_lock.release()

    def _close_session_locked(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            session.close()
