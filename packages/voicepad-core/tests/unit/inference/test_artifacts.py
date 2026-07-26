from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from voicepad_core.inference import artifacts
from voicepad_core.inference.artifacts import ArtifactError, locate_artifact, prepare_artifact
from voicepad_core.models import (
    ArtifactSource,
    DirectUrlArtifact,
    HuggingFaceArtifact,
    LocalArtifact,
    ModelSpec,
)


def _spec(source: ArtifactSource, *required_files: str) -> ModelSpec:
    return ModelSpec(
        "test-model",
        backend_id="native",
        artifact_format="gguf",
        artifact_source=source,
        required_files=required_files,
    )


class TestPrepareArtifact:
    def test_local_source_is_validated_without_copying(self, tmp_path: Path) -> None:
        """A complete local artifact is returned directly and is not installed."""
        local = tmp_path / "local"
        local.mkdir()
        (local / "model.gguf").write_bytes(b"model")

        prepared = prepare_artifact(
            _spec(LocalArtifact(local), "model.gguf"),
            tmp_path / "cache",
        )

        assert prepared == local

    def test_direct_file_is_hash_checked_and_installed_atomically(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A verified direct file appears only at its completed cache location."""
        payload = b"verified-model"
        source = DirectUrlArtifact(
            "https://example.test/model.gguf",
            sha256=hashlib.sha256(payload).hexdigest(),
        )

        def download(_: str, destination: Path) -> None:
            destination.write_bytes(payload)

        monkeypatch.setattr(artifacts, "_download_url", download)

        prepared = prepare_artifact(_spec(source, "model.gguf"), tmp_path / "cache")

        assert prepared.joinpath("model.gguf").read_bytes() == payload

    def test_direct_file_rejects_hash_mismatch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mismatched direct download is rejected before cache installation."""
        source = DirectUrlArtifact(
            "https://example.test/model.gguf",
            sha256="0" * 64,
        )
        monkeypatch.setattr(
            artifacts,
            "_download_url",
            lambda _url, destination: destination.write_bytes(b"wrong"),
        )

        with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
            prepare_artifact(_spec(source, "model.gguf"), tmp_path / "cache")

    def test_zip_archive_is_extracted_before_layout_validation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A declared ZIP archive installs its validated extracted layout."""
        source = DirectUrlArtifact(
            "https://example.test/model.zip",
            archive="zip",
            root="model-bundle",
        )

        def download(_: str, destination: Path) -> None:
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("model-bundle/weights/model.gguf", b"model")

        monkeypatch.setattr(artifacts, "_download_url", download)

        prepared = prepare_artifact(
            _spec(source, "weights/model.gguf"),
            tmp_path / "cache",
        )

        assert prepared.joinpath("weights/model.gguf").is_file()

    def test_direct_download_reports_byte_progress(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The provider exposes download byte progress without coupling to a UI."""
        source = DirectUrlArtifact("https://example.test/model.gguf")
        progress: list[tuple[int, int | None]] = []

        def download(
            _: str,
            destination: Path,
            on_progress,
        ) -> None:
            destination.write_bytes(b"model")
            on_progress(5, 5)

        monkeypatch.setattr(artifacts, "_download_url", download)

        prepare_artifact(
            _spec(source, "model.gguf"),
            tmp_path / "cache",
            on_progress=lambda downloaded, total: progress.append((downloaded, total)),
        )

        assert progress == [(5, 5)]

    def test_hugging_face_source_uses_declared_revision(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Hugging Face source is acquired through its explicit provider."""
        seen: list[HuggingFaceArtifact] = []
        source = HuggingFaceArtifact("owner/model", "commit")

        def download(declared_source: HuggingFaceArtifact, destination: Path) -> None:
            seen.append(declared_source)
            (destination / "model.bin").write_bytes(b"model")

        monkeypatch.setattr(artifacts, "_download_hugging_face", download)

        prepare_artifact(_spec(source, "model.bin"), tmp_path / "cache")

        assert seen == [source]

    def test_hugging_face_download_limits_snapshot_to_declared_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model can exclude unrelated formats from a Hugging Face snapshot."""
        calls: list[dict[str, object]] = []
        source = HuggingFaceArtifact(
            "nvidia/parakeet",
            "commit",
            allow_patterns=("model.safetensors", "config.json"),
        )

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda **kwargs: calls.append(kwargs),
        )

        artifacts._download_hugging_face(source, tmp_path)

        assert calls == [
            {
                "repo_id": "nvidia/parakeet",
                "revision": "commit",
                "local_dir": str(tmp_path),
                "allow_patterns": ["model.safetensors", "config.json"],
            }
        ]

    def test_cached_artifact_skips_provider(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid installed artifact is reused without another acquisition."""
        source = DirectUrlArtifact("https://example.test/model.gguf")
        calls = 0

        def download(_: str, destination: Path) -> None:
            nonlocal calls
            calls += 1
            destination.write_bytes(b"model")

        monkeypatch.setattr(artifacts, "_download_url", download)
        spec = _spec(source, "model.gguf")

        prepare_artifact(spec, tmp_path / "cache")
        prepare_artifact(spec, tmp_path / "cache")

        assert calls == 1


class TestLocateArtifact:
    def test_returns_valid_local_artifact_without_creating_cache(self, tmp_path: Path) -> None:
        """A local catalogue entry is checked without touching the cache."""
        local = tmp_path / "local"
        local.mkdir()
        (local / "model.gguf").write_bytes(b"model")
        cache = tmp_path / "cache"

        located = locate_artifact(_spec(LocalArtifact(local), "model.gguf"), cache)

        assert located == local
        assert not cache.exists()

    def test_returns_none_for_missing_remote_artifact(self, tmp_path: Path) -> None:
        """A remote entry is reported missing without starting a download."""
        source = DirectUrlArtifact("https://example.test/model.gguf")

        located = locate_artifact(_spec(source, "model.gguf"), tmp_path / "cache")

        assert located is None
        assert not (tmp_path / "cache").exists()

    def test_returns_cached_remote_artifact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A previously installed remote entry resolves to its validated path."""
        source = DirectUrlArtifact("https://example.test/model.gguf")
        spec = _spec(source, "model.gguf")
        monkeypatch.setattr(
            artifacts,
            "_download_url",
            lambda _url, destination: destination.write_bytes(b"model"),
        )

        prepared = prepare_artifact(spec, tmp_path / "cache")
        located = locate_artifact(spec, tmp_path / "cache")

        assert located == prepared
