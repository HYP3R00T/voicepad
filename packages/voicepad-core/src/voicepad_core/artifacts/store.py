from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Protocol

from voicepad_core.deployments import ArtifactFile, ArtifactManifest, ArtifactSource, HttpSource, HuggingFaceSource

ProgressCallback = Callable[[int, int], None]
PROVENANCE_FILE = "voicepad-artifact.json"
COPY_BUFFER_SIZE = 1024 * 1024


class ArtifactError(RuntimeError):
    """Base error for curated artifact operations."""


class ArtifactNotPreparedError(ArtifactError):
    """Raised when a requested artifact snapshot does not exist."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when artifact bytes do not match their immutable manifest."""


class ArtifactAcquisitionError(ArtifactError):
    """Raised when an immutable source cannot be acquired."""


class BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class ArtifactAcquirer(Protocol):
    def acquire(
        self,
        source: ArtifactSource,
        artifact: ArtifactFile,
        destination: Path,
        operation_dir: Path,
    ) -> None: ...


class HuggingFaceAcquirer:
    """Acquire one pinned Hugging Face file into operation-owned storage."""

    def acquire(
        self,
        source: ArtifactSource,
        artifact: ArtifactFile,
        destination: Path,
        operation_dir: Path,
    ) -> None:
        if not isinstance(source, HuggingFaceSource):
            raise ArtifactAcquisitionError("Hugging Face acquirer received a non-Hugging-Face source.")
        try:
            from huggingface_hub import hf_hub_download

            downloaded = Path(
                hf_hub_download(
                    repo_id=source.repository,
                    filename=artifact.path,
                    revision=source.revision,
                    cache_dir=operation_dir / "huggingface-cache",
                    local_files_only=False,
                )
            )
            _copy_bounded(downloaded, destination, artifact.size)
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactAcquisitionError(
                f"Could not acquire {source.repository}@{source.revision}:{artifact.path}: {error}"
            ) from error


class HttpAcquirer:
    """Download one immutable HTTPS artifact with an exact byte bound."""

    def acquire(
        self,
        source: ArtifactSource,
        artifact: ArtifactFile,
        destination: Path,
        operation_dir: Path,
    ) -> None:
        del operation_dir
        if not isinstance(source, HttpSource):
            raise ArtifactAcquisitionError("HTTP acquirer received a non-HTTP source.")
        request = urllib.request.Request(source.url, headers={"User-Agent": "VoicePad artifact preparation"})
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310 - curated HTTPS URL
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != artifact.size:
                    raise ArtifactIntegrityError(f"HTTP artifact size does not match: {artifact.path}")
                _write_stream_bounded(response, destination, artifact.size)
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactAcquisitionError(f"Could not acquire {source.url}: {error}") from error


class CuratedAcquirer:
    def __init__(self) -> None:
        self._hugging_face = HuggingFaceAcquirer()
        self._http = HttpAcquirer()

    def acquire(
        self,
        source: ArtifactSource,
        artifact: ArtifactFile,
        destination: Path,
        operation_dir: Path,
    ) -> None:
        if isinstance(source, HuggingFaceSource):
            self._hugging_face.acquire(source, artifact, destination, operation_dir)
            return
        self._http.acquire(source, artifact, destination, operation_dir)


class ArtifactStore:
    """Prepare and verify curated artifact snapshots."""

    def __init__(self, root: Path, acquirer: ArtifactAcquirer | None = None) -> None:
        self.root = root.expanduser().resolve()
        self._snapshots = self.root / "snapshots"
        self._acquirer = acquirer or CuratedAcquirer()
        self._lock = RLock()

    def snapshot_path(self, manifest: ArtifactManifest) -> Path:
        return self._snapshots / manifest.id

    def locate(self, manifest: ArtifactManifest) -> Path | None:
        target = self.snapshot_path(manifest)
        if not os.path.lexists(target):
            return None
        return self.verify(manifest)

    def verify(self, manifest: ArtifactManifest) -> Path:
        target = self.snapshot_path(manifest)
        if not target.is_dir():
            raise ArtifactNotPreparedError(f"Artifact snapshot is not prepared: {manifest.id}")

        for artifact in manifest.files:
            _verify_file(target / artifact.path, artifact)

        provenance = target / PROVENANCE_FILE
        if not provenance.is_file() or provenance.is_symlink():
            raise ArtifactIntegrityError(f"Artifact snapshot has no trusted provenance: {manifest.id}")
        try:
            recorded = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(f"Artifact snapshot provenance is invalid: {manifest.id}") from error
        if recorded != _provenance(manifest):
            raise ArtifactIntegrityError(f"Artifact snapshot provenance does not match: {manifest.id}")
        return target

    def prepare(
        self,
        manifest: ArtifactManifest,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        with self._lock:
            existing = self.locate(manifest)
            if existing is not None:
                return existing

            self._snapshots.mkdir(parents=True, exist_ok=True)
            operation = Path(tempfile.mkdtemp(prefix=f".{manifest.id}-", dir=self.root))
            snapshot = operation / "snapshot"
            snapshot.mkdir()
            completed = 0
            try:
                for artifact in manifest.files:
                    destination = snapshot / artifact.path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self._acquirer.acquire(manifest.source, artifact, destination, operation)
                    _verify_file(destination, artifact)
                    completed += artifact.size
                    if on_progress is not None:
                        on_progress(completed, manifest.total_size)

                _write_provenance(snapshot / PROVENANCE_FILE, manifest)
                _flush_tree(snapshot)
                target = self.snapshot_path(manifest)
                if os.path.lexists(target):
                    return self.verify(manifest)
                os.replace(snapshot, target)
                _flush_directory(self._snapshots)
                return self.verify(manifest)
            except ArtifactError:
                raise
            except Exception as error:
                raise ArtifactAcquisitionError(f"Could not prepare artifact snapshot {manifest.id}: {error}") from error
            finally:
                shutil.rmtree(operation, ignore_errors=True)


def _copy_bounded(source: Path, destination: Path, maximum_size: int) -> None:
    with source.open("rb") as input_file:
        _write_stream_bounded(input_file, destination, maximum_size)


def _write_stream_bounded(input_file: BinaryReader, destination: Path, maximum_size: int) -> None:
    copied = 0
    with destination.open("xb") as output_file:
        while chunk := input_file.read(COPY_BUFFER_SIZE):
            copied += len(chunk)
            if copied > maximum_size:
                raise ArtifactIntegrityError(f"Artifact exceeds declared size: {destination.name}")
            output_file.write(chunk)
        if copied != maximum_size:
            raise ArtifactIntegrityError(f"Artifact size does not match: {destination.name}")
        output_file.flush()
        os.fsync(output_file.fileno())


def _verify_file(path: Path, artifact: ArtifactFile) -> None:
    if not path.is_file() or path.is_symlink():
        raise ArtifactIntegrityError(f"Required artifact is missing or unsafe: {artifact.path}")
    if path.stat().st_size != artifact.size:
        raise ArtifactIntegrityError(f"Artifact size does not match: {artifact.path}")

    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(COPY_BUFFER_SIZE):
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        raise ArtifactIntegrityError(f"Artifact SHA-256 does not match: {artifact.path}")


def _provenance(manifest: ArtifactManifest) -> dict[str, object]:
    if isinstance(manifest.source, HuggingFaceSource):
        source: dict[str, object] = {
            "kind": "huggingface",
            "repository": manifest.source.repository,
            "revision": manifest.source.revision,
        }
    else:
        source = {"kind": "https", "url": manifest.source.url}
    return {
        "schema": 1,
        "manifest_id": manifest.id,
        "source": source,
        "license": manifest.license,
        "files": [
            {"path": artifact.path, "size": artifact.size, "sha256": artifact.sha256} for artifact in manifest.files
        ],
    }


def _write_provenance(path: Path, manifest: ArtifactManifest) -> None:
    with path.open("x", encoding="utf-8") as provenance_file:
        json.dump(_provenance(manifest), provenance_file, indent=2, sort_keys=True)
        provenance_file.write("\n")
        provenance_file.flush()
        os.fsync(provenance_file.fileno())


def _flush_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as file:
                os.fsync(file.fileno())
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        _flush_directory(directory)


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
