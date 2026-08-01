from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import urlparse


class Precision(StrEnum):
    FP16 = "fp16"


class TimestampGranularity(StrEnum):
    TOKEN_DURATION = "token-duration"


class ContextBiasingMode(StrEnum):
    NONE = "none"


class CancellationMode(StrEnum):
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.path:
            raise ValueError(f"artifact path must be a normalized relative POSIX path: {self.path!r}")
        if self.size <= 0:
            raise ValueError("artifact size must be positive")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("artifact SHA-256 must contain 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class HuggingFaceSource:
    repository: str
    revision: str

    def __post_init__(self) -> None:
        if self.repository.count("/") != 1:
            raise ValueError("Hugging Face repository must use the 'owner/name' form")
        if len(self.revision) != 40 or any(character not in "0123456789abcdef" for character in self.revision):
            raise ValueError("Hugging Face revision must be a full lowercase commit SHA")


@dataclass(frozen=True, slots=True)
class HttpSource:
    url: str

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("artifact URL must be an unauthenticated absolute HTTPS URL")


ArtifactSource = HuggingFaceSource | HttpSource


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    id: str
    source: ArtifactSource
    license: str
    files: tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("artifact manifest ID must not be empty")
        if not self.files:
            raise ValueError("artifact manifest must contain at least one file")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact manifest paths must be unique")

    @property
    def total_size(self) -> int:
        return sum(file.size for file in self.files)


@dataclass(frozen=True, slots=True)
class WheelExtraction:
    id: str
    wheel_manifest_id: str
    entry_path: str
    output_name: str
    size: int
    compressed_size: int
    sha256: str

    def __post_init__(self) -> None:
        ArtifactFile(self.entry_path, self.size, self.sha256)
        output = PurePosixPath(self.output_name)
        if output.is_absolute() or len(output.parts) != 1 or str(output) != self.output_name:
            raise ValueError("extracted artifact output must be one normalized filename")
        if self.compressed_size <= 0:
            raise ValueError("compressed artifact size must be positive")


@dataclass(frozen=True, slots=True)
class DeclaredCapabilities:
    native_sample_rate: int
    languages: tuple[str, ...]
    timestamps: TimestampGranularity
    accepts_language_hint: bool = False
    returns_detected_language: bool = False
    native_streaming: bool = False
    translation_targets: tuple[str, ...] = ()
    context_biasing: ContextBiasingMode = ContextBiasingMode.NONE
    cancellation: CancellationMode = CancellationMode.GENERATION


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    required_device: str
    platforms: tuple[str, ...]
    minimum_gpu_memory_bytes: int
    measured_gpu: str
    measured_peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class ProcessingProfile:
    preferred_chunk_seconds: int
    maximum_input_seconds: int
    warmup_seconds: int


@dataclass(frozen=True, slots=True)
class DeploymentDefinition:
    id: str
    model_id: str
    artifact_manifest_id: str
    adapter_id: str
    precision: Precision
    capabilities: DeclaredCapabilities
    resources: ResourceProfile
    processing: ProcessingProfile
    recommended: bool
