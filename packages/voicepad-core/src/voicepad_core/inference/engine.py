from __future__ import annotations

import logging
import time
from collections.abc import Callable
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


class ResidentTranscriptionEngine:
    """Own one warmed deployment session and serialize all model operations."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        session_factory: SessionFactory = TransformersParakeetTDTSession,
        device_admitter: DeviceAdmitter = admit_cuda_device,
        artifact_store: ArtifactProvider | None = None,
    ) -> None:
        self._store = artifact_store or ArtifactStore(artifact_root)
        self._session_factory = session_factory
        self._admit_device = device_admitter
        self._session: TranscriptionSession | None = None
        self._state_lock = RLock()
        self._operation_lock = Lock()

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
                if self._session is not None and self._session.deployment.definition.id == deployment_id:
                    logger.info("Deployment already active: deployment=%s", deployment_id)
                    return self._session.deployment
                self._close_session_locked()

            snapshot = self._store.prepare(manifest, on_progress)
            device = self._admit_device(definition, device_index)
            candidate = self._session_factory(definition, manifest, snapshot, device)
            candidate.warm()
            with self._state_lock:
                self._session = candidate
                candidate = None
                logger.info(
                    "Deployment activation finished: deployment=%s device=%s precision=%s latency_s=%.3f",
                    self._session.deployment.definition.id,
                    self._session.deployment.device_name,
                    self._session.deployment.definition.precision,
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
                if self._session is None:
                    raise InferenceError("The transcription engine is not ready.")
                session = self._session
            try:
                result = session.transcribe(
                    audio,
                    intent or TranscriptionIntent(),
                    cancellation or CancellationToken(),
                )
            except (InvalidTranscriptionInputError, UnsupportedIntentError):
                raise
            except Exception as error:
                with self._state_lock:
                    try:
                        self._close_session_locked()
                    except Exception as cleanup_error:
                        logger.exception("Session cleanup failed after transcription failure")
                        error.add_note(f"Session cleanup also failed: {cleanup_error}")
                raise
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
                try:
                    self._close_session_locked()
                except Exception:
                    logger.exception("Session cleanup failed while unloading")
                    raise
                logger.info("Deployment unloaded: deployment=%s", deployment_id or "none")
        finally:
            self._operation_lock.release()

    def _close_session_locked(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            session.close()
