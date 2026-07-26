from pathlib import Path

import pytest
from voicepad_core.inference.artifacts import ArtifactError, artifact_path, locate_artifact, prepare_artifact
from voicepad_core.models import Model


def _model() -> Model:
    return Model("tiny", "owner/tiny", "test", ("model.bin",), "Tiny", "Test model", revision="commit")


def test_artifact_path_groups_models_by_backend(tmp_path: Path) -> None:
    """Cached models have one predictable directory."""
    assert artifact_path(_model(), tmp_path) == tmp_path / "test" / "tiny"


def test_locate_artifact_returns_none_for_incomplete_cache(tmp_path: Path) -> None:
    """An incomplete cached model is not reported as ready."""
    assert locate_artifact(_model(), tmp_path) is None


def test_prepare_downloads_declared_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation forwards the repository, revision, and file list."""
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> None:
        calls.append(kwargs)
        (Path(str(kwargs["local_dir"])) / "model.bin").write_bytes(b"model")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    result = prepare_artifact(_model(), tmp_path)

    assert result == tmp_path / "test" / "tiny"
    assert calls == [
        {
            "repo_id": "owner/tiny",
            "revision": "commit",
            "local_dir": str(result),
            "allow_patterns": ["model.bin"],
        }
    ]


def test_prepare_wraps_download_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider failures are reported with model context."""

    def fail(**_: object) -> None:
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    with pytest.raises(ArtifactError, match="tiny.*offline"):
        prepare_artifact(_model(), tmp_path)
