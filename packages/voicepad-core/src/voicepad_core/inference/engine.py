from __future__ import annotations

import logging
import time
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
logger = logging.getLogger(__name__)


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
        started = time.perf_counter()
        logger.info("Deployment activation started: deployment=%s device_index=%s", deployment_id, device_index)
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
                    logger.info("Deployment already active: deployment=%s", deployment_id)
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
                logger.info(
                    "Deployment activation finished: deployment=%s device=%s precision=%s latency_s=%.3f",
                    self._session.deployment.definition.id,
                    self._session.deployment.device_name,
                    self._session.deployment.definition.precision.value,
                    time.perf_counter() - started,
                )
                return self._session.deployment
        except Exception as error:
            if candidate is not None:
                try:
                    candidate.close()
                except Exception as cleanup_error:
                    logger.exception("Session cleanup failed after activation failure")
                    error.add_note(f"Session cleanup also failed: {cleanup_error}")
            with self._state_lock:
                self._session = None
                self._state = EngineState.FAILED
            logger.error(
                "Deployment activation failed: deployment=%s error_type=%s error=%s",
                deployment_id,
                type(error).__name__,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
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
        started = time.perf_counter()
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
            except Exception as error:
                with self._state_lock:
                    try:
                        self._close_session_locked()
                    except Exception as cleanup_error:
                        logger.exception("Session cleanup failed after transcription failure")
                        error.add_note(f"Session cleanup also failed: {cleanup_error}")
                    finally:
                        self._state = EngineState.FAILED
                raise
            with self._state_lock:
                self._state = EngineState.READY
            logger.info(
                "Inference finished: deployment=%s samples=%s sample_rate=%s tokens=%s words=%s "
                "cancelled=%s latency_s=%.3f",
                session.deployment.definition.id,
                len(audio.samples),
                audio.sample_rate,
                len(result.tokens),
                len(result.words),
                result.cancelled,
                time.perf_counter() - started,
            )
            return result
        finally:
            self._operation_lock.release()

    def unload(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeBusyError("The transcription engine is busy.")
        try:
            with self._state_lock:
                deployment_id = self._session.deployment.definition.id if self._session is not None else None
                self._state = EngineState.UNLOADING
                try:
                    self._close_session_locked()
                except Exception:
                    self._state = EngineState.FAILED
                    logger.exception("Session cleanup failed while unloading")
                    raise
                self._state = EngineState.UNPREPARED
                logger.info("Deployment unloaded: deployment=%s", deployment_id or "none")
        finally:
            self._operation_lock.release()

    def _close_session_locked(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            session.close()
