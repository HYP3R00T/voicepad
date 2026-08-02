from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

_RELEASE_PATH = Path(__file__).parents[4] / "scripts" / "release.py"
_RELEASE_SPEC = importlib.util.spec_from_file_location("voicepad_release_workflow", _RELEASE_PATH)
if _RELEASE_SPEC is None or _RELEASE_SPEC.loader is None:
    raise RuntimeError(f"could not load release helper from {_RELEASE_PATH}")
release = importlib.util.module_from_spec(_RELEASE_SPEC)
sys.modules[_RELEASE_SPEC.name] = release
_RELEASE_SPEC.loader.exec_module(release)


def _write_package(root: Path, package: str, version: str, dependencies: list[str] | None = None) -> None:
    package_dir = root / "packages" / package
    package_dir.mkdir(parents=True)
    dependency_text = ""
    if dependencies is not None:
        items = "\n".join(f'  "{item}",' for item in dependencies)
        dependency_text = f"dependencies = [\n{items}\n]\n"
    (package_dir / "pyproject.toml").write_text(
        f'[project]\nname = "{package}"\nversion = "{version}"\n{dependency_text}'
    )


def test_validate_input_accepts_every_browser_choice() -> None:
    for package in release.PACKAGES:
        for bump in release.BUMPS:
            release.validate_input(package, bump)


@pytest.mark.parametrize(
    ("package", "bump"),
    [("other", "patch"), ("voicepad", "other"), ("voicepad; echo unsafe", "minor")],
)
def test_validate_input_rejects_values_outside_browser_choices(package: str, bump: str) -> None:
    with pytest.raises(release.ReleaseError):
        release.validate_input(package, bump)


@pytest.mark.parametrize(
    ("branch", "package", "version", "prerelease"),
    [
        ("release/voicepad-v0.3.0", "voicepad", "0.3.0", False),
        ("release/voicepad-core-v1.2.3rc1", "voicepad-core", "1.2.3rc1", True),
        ("release/voicepad-v1.2.3.dev4", "voicepad", "1.2.3.dev4", True),
        ("release/voicepad-v1.2.3.post1", "voicepad", "1.2.3.post1", False),
    ],
)
def test_parse_release_branch(branch: str, package: str, version: str, prerelease: bool) -> None:
    parsed = release.parse_release_branch(branch)
    assert parsed.package == package
    assert parsed.version == version
    assert parsed.branch == branch
    assert parsed.tag == f"{package}-v{version}"
    assert parsed.is_prerelease is prerelease


@pytest.mark.parametrize(
    "branch",
    [
        "release/voicepad-v0.3",
        "release/voicepad-v0.3.0/extra",
        "release/other-v0.3.0",
        "feature/voicepad-v0.3.0",
        "release/voicepad-v0.3.0;echo-unsafe",
    ],
)
def test_parse_release_branch_rejects_unrecognized_refs(branch: str) -> None:
    with pytest.raises(release.ReleaseError):
        release.parse_release_branch(branch)


def test_validate_release_version_matches_reviewed_metadata(tmp_path: Path) -> None:
    _write_package(tmp_path, "voicepad", "0.3.0")
    current = release.Release("voicepad", "0.3.0")
    release.validate_release_version(tmp_path, current)

    with pytest.raises(release.ReleaseError, match="does not match"):
        release.validate_release_version(tmp_path, release.Release("voicepad", "0.3.1"))


def test_validate_changed_files_accepts_only_release_allowlist() -> None:
    current = release.Release("voicepad-core", "0.3.0")
    release.validate_changed_files(
        current,
        {
            "packages/voicepad-core/CHANGELOG.md",
            "packages/voicepad-core/pyproject.toml",
            "requirements.txt",
            "uv.lock",
        },
    )


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"packages/voicepad/pyproject.toml"}, "missing required"),
        (
            {
                "packages/voicepad/CHANGELOG.md",
                "packages/voicepad/pyproject.toml",
                "packages/voicepad/src/voicepad/main.py",
            },
            "unexpected files",
        ),
    ],
)
def test_validate_changed_files_fails_closed(files: set[str], message: str) -> None:
    with pytest.raises(release.ReleaseError, match=message):
        release.validate_changed_files(release.Release("voicepad", "0.3.0"), files)


def test_sync_core_dependency_uses_exact_local_core_version(tmp_path: Path) -> None:
    _write_package(tmp_path, "voicepad-core", "0.3.0")
    _write_package(tmp_path, "voicepad", "0.4.0", ["textual>=8", "voicepad-core>=0.2.1"])

    assert release.sync_core_dependency(tmp_path)
    assert release.validate_core_dependency(tmp_path) == "0.3.0"
    assert not release.sync_core_dependency(tmp_path)


def test_validate_core_dependency_rejects_stale_or_ambiguous_requirement(tmp_path: Path) -> None:
    _write_package(tmp_path, "voicepad-core", "0.3.0")
    _write_package(tmp_path, "voicepad", "0.4.0", ["voicepad-core>=0.2.1"])

    with pytest.raises(release.ReleaseError, match="must be exactly"):
        release.validate_core_dependency(tmp_path)


def test_artifact_digests_accepts_only_isolated_distribution_files(tmp_path: Path) -> None:
    wheel = tmp_path / "voicepad-0.3.0-py3-none-any.whl"
    source = tmp_path / "voicepad-0.3.0.tar.gz"
    wheel.write_bytes(b"wheel")
    source.write_bytes(b"source")
    (tmp_path / ".gitignore").write_text("*")

    assert release.artifact_digests(tmp_path) == {
        wheel.name: hashlib.sha256(b"wheel").hexdigest(),
        source.name: hashlib.sha256(b"source").hexdigest(),
    }

    (tmp_path / "unexpected.txt").write_text("unexpected")
    with pytest.raises(release.ReleaseError, match="unexpected release artifacts"):
        release.artifact_digests(tmp_path)


def test_artifact_digests_requires_one_wheel_and_one_source_distribution(tmp_path: Path) -> None:
    (tmp_path / "voicepad-0.3.0.tar.gz").write_bytes(b"source")

    with pytest.raises(release.ReleaseError, match="exactly one wheel"):
        release.artifact_digests(tmp_path)


def test_validate_artifact_names_matches_normalized_package_and_version() -> None:
    release.validate_artifact_names(
        {
            "voicepad_core-0.3.0-py3-none-any.whl",
            "voicepad_core-0.3.0.tar.gz",
        },
        "voicepad-core",
        "0.3.0",
    )

    with pytest.raises(release.ReleaseError, match="do not match"):
        release.validate_artifact_names(
            {
                "other-0.3.0-py3-none-any.whl",
                "other-0.3.0.tar.gz",
            },
            "voicepad-core",
            "0.3.0",
        )


def test_compare_pypi_artifacts_distinguishes_resume_and_conflict() -> None:
    local = {
        "voicepad-0.3.0-py3-none-any.whl": "def",
        "voicepad-0.3.0.tar.gz": "abc",
    }
    partial = {"voicepad-0.3.0.tar.gz": "abc"}
    assert release.compare_pypi_artifacts(local, None) == "absent"
    assert release.compare_pypi_artifacts(local, partial) == "partial"
    assert release.compare_pypi_artifacts(local, local.copy()) == "matching"
    assert release.compare_pypi_artifacts(local, {"voicepad-0.3.0.tar.gz": "different"}) == "conflict"
    assert release.missing_pypi_artifacts(local, partial) == ["voicepad-0.3.0-py3-none-any.whl"]


def test_missing_pypi_artifacts_rejects_unexpected_remote_file() -> None:
    with pytest.raises(release.ReleaseError, match="conflicting artifacts"):
        release.missing_pypi_artifacts(
            {"voicepad-0.3.0.tar.gz": "abc"},
            {"different-project-0.3.0.tar.gz": "abc"},
        )


def test_fetch_pypi_artifacts_reads_public_sha256_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "urls": [
            {
                "filename": "voicepad-0.3.0.tar.gz",
                "digests": {"sha256": "abc123"},
            }
        ]
    }
    monkeypatch.setattr(
        release.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(json.dumps(payload).encode())
    )

    assert release.fetch_pypi_artifacts("voicepad", "0.3.0") == {"voicepad-0.3.0.tar.gz": "abc123"}


def test_fetch_pypi_artifacts_treats_404_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(*_args: object, **_kwargs: object) -> None:
        raise HTTPError("https://pypi.org", 404, "not found", Message(), None)

    monkeypatch.setattr(release.urllib.request, "urlopen", not_found)

    assert release.fetch_pypi_artifacts("voicepad-core", "0.3.0") is None
