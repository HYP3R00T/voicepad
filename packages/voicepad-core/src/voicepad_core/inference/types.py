from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Protocol

from voicepad_core.deployments import DeclaredCapabilities, DeploymentDefinition
from voicepad_core.preprocessing import PreprocessedAudio


class InferenceError(RuntimeError):
    """Base error for transcription runtime failures."""


class UnsupportedPlatformError(InferenceError):
    """Raised when a deployment is unavailable on the current platform."""


class CudaAdmissionError(InferenceError):
    """Raised when the selected NVIDIA device cannot safely run a deployment."""


class UnsupportedIntentError(InferenceError):
    """Raised when a request uses a capability the deployment does not expose."""


class InvalidTranscriptionInputError(InferenceError):
    """Raised when audio violates the selected deployment contract."""


class RuntimeBusyError(InferenceError):
    """Raised when an operation conflicts with active inference."""


class SessionClosedError(InferenceError):
    """Raised when work is submitted to a closed session."""


@dataclass(frozen=True, slots=True)
class TranscriptionIntent:
    language: str | None = None
    vocabulary: tuple[str, ...] = ()


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


@dataclass(frozen=True, slots=True)
class ActiveDeployment:
    definition: DeploymentDefinition
    snapshot_revision: str
    device_id: str
    device_name: str
    total_gpu_memory_bytes: int


@dataclass(frozen=True, slots=True)
class TokenTimestamp:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class BackendResult:
    text: str
    tokens: tuple[TokenTimestamp, ...]
    words: tuple[TimedWord, ...]
    cancelled: bool = False


class TranscriptionSession(Protocol):
    @property
    def deployment(self) -> ActiveDeployment: ...

    @property
    def capabilities(self) -> DeclaredCapabilities: ...

    def transcribe(
        self,
        audio: PreprocessedAudio,
        intent: TranscriptionIntent,
        cancellation: CancellationToken,
    ) -> BackendResult: ...

    def close(self) -> None: ...
