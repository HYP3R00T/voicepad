from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from ..models import (
    DirectUrlArtifact,
    HuggingFaceArtifact,
    LocalArtifact,
    ModelCompatibilityError,
    ModelSpec,
    validate_model_artifact,
)


class ArtifactError(RuntimeError):
    """Raised when a declared model artifact cannot be prepared."""


ProgressCallback = Callable[[int, int | None], None]

_install_locks_guard = Lock()
_install_locks: dict[Path, Lock] = {}


def locate_artifact(model: ModelSpec, cache_dir: Path) -> Path | None:
    """Return a validated local artifact without acquiring anything."""
    source = model.artifact_source
    if source is None:
        return None

    if isinstance(source, LocalArtifact):
        local_path = source.path.expanduser().resolve()
        artifact_root = local_path.parent if local_path.is_file() else local_path
    else:
        artifact_root = _remote_target(model, source, cache_dir)

    try:
        return validate_model_artifact(artifact_root, model)
    except ModelCompatibilityError:
        return None


def prepare_artifact(
    model: ModelSpec,
    cache_dir: Path,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Resolve, acquire, validate, and atomically install a model artifact."""
    source = model.artifact_source
    if source is None:
        raise ArtifactError(f"Model '{model.id}' does not declare an artifact source.")

    if isinstance(source, LocalArtifact):
        local_path = source.path.expanduser().resolve()
        artifact_root = local_path.parent if local_path.is_file() else local_path
        return validate_model_artifact(artifact_root, model)

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _remote_target(model, source, cache_dir)

    with _install_lock(target):
        return _prepare_remote_artifact(model, source, target, on_progress)


def _prepare_remote_artifact(
    model: ModelSpec,
    source: HuggingFaceArtifact | DirectUrlArtifact,
    target: Path,
    on_progress: ProgressCallback | None,
) -> Path:
    if target.exists():
        try:
            return validate_model_artifact(target, model)
        except ModelCompatibilityError as exc:
            raise ArtifactError(f"Cached artifact for model '{model.id}' is incomplete: {exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{_safe_component(model.id)}-",
            dir=target.parent,
        )
    )
    artifact_root = staging / "artifact"
    artifact_root.mkdir()

    try:
        if isinstance(source, HuggingFaceArtifact):
            _download_hugging_face(source, artifact_root)
            if on_progress is not None:
                installed_bytes = sum(path.stat().st_size for path in artifact_root.rglob("*") if path.is_file())
                on_progress(installed_bytes, installed_bytes)
        elif isinstance(source, DirectUrlArtifact):
            _download_direct(source, artifact_root, staging, on_progress)
        else:
            raise ArtifactError(f"Model '{model.id}' uses an unsupported artifact source: {type(source).__name__}.")

        install_root = artifact_root
        if isinstance(source, DirectUrlArtifact) and source.root is not None:
            install_root = artifact_root / source.root
        validate_model_artifact(install_root, model)
        try:
            os.replace(install_root, target)
        except OSError as exc:
            if target.exists():
                return validate_model_artifact(target, model)
            raise ArtifactError(f"Could not install artifact for model '{model.id}': {exc}") from exc
        return target
    except (ArtifactError, ModelCompatibilityError):
        raise
    except Exception as exc:
        raise ArtifactError(f"Could not prepare artifact for model '{model.id}': {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _download_hugging_face(
    source: HuggingFaceArtifact,
    destination: Path,
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=source.repo_id,
        revision=source.revision,
        local_dir=str(destination),
    )


def _download_direct(
    source: DirectUrlArtifact,
    destination: Path,
    staging: Path,
    on_progress: ProgressCallback | None,
) -> None:
    download_path = staging / "download"
    if on_progress is None:
        _download_url(source.url, download_path)
    else:
        _download_url(source.url, download_path, on_progress)
    if source.sha256 is not None:
        actual_digest = _sha256(download_path)
        if actual_digest.lower() != source.sha256.lower():
            raise ArtifactError(f"SHA-256 mismatch for '{source.url}': expected {source.sha256}, got {actual_digest}.")

    if source.archive == "zip":
        _extract_zip(download_path, destination)
    elif source.archive == "tar":
        _extract_tar(download_path, destination)
    else:
        filename = source.filename or Path(urlparse(source.url).path).name or "artifact.bin"
        filename_path = Path(filename)
        if filename_path.is_absolute() or len(filename_path.parts) != 1 or ".." in filename_path.parts:
            raise ArtifactError(f"Direct artifact filename is unsafe: {filename!r}.")
        shutil.copyfile(download_path, destination / filename)


def _download_url(
    url: str,
    destination: Path,
    on_progress: ProgressCallback | None = None,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "VoicePad/0.2 (+https://github.com/HYP3R00T/voicepad)"},
    )
    with (
        urllib.request.urlopen(request) as response,  # noqa: S310 - URL is an explicit catalogue declaration.
        destination.open("wb") as output,
    ):
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        downloaded = 0
        while block := response.read(1024 * 1024):
            output.write(block)
            downloaded += len(block)
            if on_progress is not None:
                on_progress(downloaded, total)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        _validate_archive_members(destination, archive.namelist())
        archive.extractall(destination)


def _extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path) as archive:
        _validate_archive_members(destination, (member.name for member in archive.getmembers()))
        archive.extractall(destination, filter="data")


def _validate_archive_members(destination: Path, members: Iterable[str]) -> None:
    for member in members:
        member_path = Path(str(member))
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ArtifactError(f"Archive contains an unsafe path: {member!s}")
        if not (destination / member_path).resolve().is_relative_to(destination.resolve()):
            raise ArtifactError(f"Archive contains an unsafe path: {member!s}")


def _safe_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "-" for character in value)
    return safe.strip(".-") or "model"


def _source_key(source: HuggingFaceArtifact | DirectUrlArtifact) -> str:
    identity = repr(source).encode()
    return hashlib.sha256(identity).hexdigest()[:16]


def _remote_target(
    model: ModelSpec,
    source: HuggingFaceArtifact | DirectUrlArtifact,
    cache_dir: Path,
) -> Path:
    return (
        cache_dir.expanduser().resolve()
        / _safe_component(model.backend_id)
        / _safe_component(model.id)
        / _source_key(source)
    )


def _install_lock(target: Path) -> Lock:
    with _install_locks_guard:
        return _install_locks.setdefault(target, Lock())


__all__ = [
    "ArtifactError",
    "ProgressCallback",
    "locate_artifact",
    "prepare_artifact",
]
