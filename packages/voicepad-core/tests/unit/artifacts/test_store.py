from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from voicepad_core.artifacts import ArtifactIntegrityError, ArtifactStore
from voicepad_core.deployments import ArtifactFile, ArtifactManifest, HuggingFaceSource

CONTENT = b"official immutable artifact"


class FakeAcquirer:
    def __init__(self, content: bytes = CONTENT) -> None:
        self.content = content
        self.calls = 0

    def acquire(
        self,
        source: HuggingFaceSource,
        artifact: ArtifactFile,
        destination: Path,
        operation_dir: Path,
    ) -> None:
        self.calls += 1
        assert source.revision == "a" * 40
        assert operation_dir.parent.name == "cache"
        destination.write_bytes(self.content)


def manifest() -> ArtifactManifest:
    return ArtifactManifest(
        id="test-official-artifact",
        source=HuggingFaceSource("vendor/model", "a" * 40),
        license="test-license",
        files=(ArtifactFile("nested/model.bin", len(CONTENT), hashlib.sha256(CONTENT).hexdigest()),),
    )


def test_prepare_verifies_and_atomically_promotes_snapshot(tmp_path: Path) -> None:
    acquirer = FakeAcquirer()
    store = ArtifactStore(tmp_path / "cache", acquirer)
    progress: list[tuple[int, int]] = []

    result = store.prepare(manifest(), lambda current, total: progress.append((current, total)))

    assert result == tmp_path / "cache" / "snapshots" / "test-official-artifact"
    assert (result / "nested/model.bin").read_bytes() == CONTENT
    assert store.locate(manifest()) == result
    assert progress == [(len(CONTENT), len(CONTENT))]
    assert not list((tmp_path / "cache").glob(".test-official-artifact-*"))


def test_prepare_reuses_verified_snapshot_without_acquisition(tmp_path: Path) -> None:
    acquirer = FakeAcquirer()
    store = ArtifactStore(tmp_path / "cache", acquirer)

    first = store.prepare(manifest())
    second = store.prepare(manifest())

    assert first == second
    assert acquirer.calls == 1


def test_prepare_does_not_publish_failed_acquisition(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cache", FakeAcquirer(b"wrong artifact bytes"))

    with pytest.raises(ArtifactIntegrityError, match="size does not match"):
        store.prepare(manifest())

    assert not store.snapshot_path(manifest()).exists()
    assert not list((tmp_path / "cache").glob(".test-official-artifact-*"))


def test_corrupt_published_snapshot_is_preserved_and_rejected(tmp_path: Path) -> None:
    acquirer = FakeAcquirer()
    store = ArtifactStore(tmp_path / "cache", acquirer)
    target = store.snapshot_path(manifest())
    target.mkdir(parents=True)
    corrupt = target / "nested/model.bin"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"private existing bytes")

    with pytest.raises(ArtifactIntegrityError, match="size does not match"):
        store.prepare(manifest())

    assert corrupt.read_bytes() == b"private existing bytes"
    assert acquirer.calls == 0


def test_verify_rejects_changed_provenance(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cache", FakeAcquirer())
    target = store.prepare(manifest())
    provenance = target / "voicepad-artifact.json"
    provenance.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="provenance does not match"):
        store.verify(manifest())
