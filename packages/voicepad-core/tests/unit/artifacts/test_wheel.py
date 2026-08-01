from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from voicepad_core.artifacts import ArtifactIntegrityError, ArtifactStore, WheelExtractor, wheel
from voicepad_core.deployments import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactSource,
    HuggingFaceSource,
    WheelExtraction,
)

MODEL = b"official model data" * 100
ENTRY = "package/data/model.onnx"


class WheelAcquirer:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def acquire(
        self,
        source: ArtifactSource,
        artifact: ArtifactFile,
        destination: Path,
        operation_dir: Path,
    ) -> None:
        destination.write_bytes(self.data)


def wheel_bytes(*, unsafe: bool = False) -> tuple[bytes, int]:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ENTRY, MODEL)
        if unsafe:
            archive.writestr("../unsafe", b"ignored")
    data = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        compressed_size = archive.getinfo(ENTRY).compress_size
    return data, compressed_size


def definitions(data: bytes, compressed_size: int) -> tuple[ArtifactManifest, WheelExtraction]:
    manifest = ArtifactManifest(
        "test-wheel",
        HuggingFaceSource("vendor/package", "a" * 40),
        "MIT",
        (ArtifactFile("package.whl", len(data), hashlib.sha256(data).hexdigest()),),
    )
    extraction = WheelExtraction(
        "test-model",
        manifest.id,
        ENTRY,
        "model.onnx",
        len(MODEL),
        compressed_size,
        hashlib.sha256(MODEL).hexdigest(),
    )
    return manifest, extraction


def test_wheel_extractor_publishes_only_verified_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, compressed_size = wheel_bytes()
    manifest, extraction = definitions(data, compressed_size)
    store = ArtifactStore(tmp_path / "artifacts", WheelAcquirer(data))
    monkeypatch.setattr(wheel, "get_manifest", lambda manifest_id: manifest)

    result = WheelExtractor(store).prepare(extraction)

    assert result.read_bytes() == MODEL
    assert result.name == "model.onnx"
    assert WheelExtractor(store).verify(extraction) == result


def test_wheel_extractor_rejects_unsafe_archive_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, compressed_size = wheel_bytes(unsafe=True)
    manifest, extraction = definitions(data, compressed_size)
    store = ArtifactStore(tmp_path / "artifacts", WheelAcquirer(data))
    monkeypatch.setattr(wheel, "get_manifest", lambda manifest_id: manifest)

    with pytest.raises(ArtifactIntegrityError, match="unsafe path"):
        WheelExtractor(store).prepare(extraction)

    assert not WheelExtractor(store).output_path(extraction).exists()
