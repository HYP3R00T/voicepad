"""Validation helpers for the reviewed package release workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PACKAGES: Final = ("voicepad-core", "voicepad")
BUMPS: Final = ("patch", "minor", "major", "stable", "alpha", "beta", "rc", "post", "dev")
_VERSION = r"(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*)){2}(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?"
_RELEASE_BRANCH = re.compile(rf"release/(?P<package>{'|'.join(PACKAGES)})-v(?P<version>{_VERSION})")
_CORE_DEPENDENCY = re.compile(r'^(?P<indent>\s*)"voicepad-core>=[^"]+",(?P<suffix>\s*)$', re.MULTILINE)


class ReleaseError(ValueError):
    """Raised when release state does not match the reviewed workflow contract."""


@dataclass(frozen=True, slots=True)
class Release:
    """An exact package release derived from a generated branch."""

    package: str
    version: str

    @property
    def branch(self) -> str:
        return f"release/{self.package}-v{self.version}"

    @property
    def tag(self) -> str:
        return f"{self.package}-v{self.version}"

    @property
    def package_dir(self) -> Path:
        return Path("packages") / self.package

    @property
    def is_prerelease(self) -> bool:
        return bool(re.search(r"(?:a|b|rc)\d+|\.dev\d+", self.version))


def validate_input(package: str, bump: str) -> None:
    """Reject values outside the workflow's fixed browser choices."""
    if package not in PACKAGES:
        raise ReleaseError(f"unsupported package: {package}")
    if bump not in BUMPS:
        raise ReleaseError(f"unsupported version bump: {bump}")


def parse_release_branch(branch: str) -> Release:
    """Parse an exact generated release branch name."""
    match = _RELEASE_BRANCH.fullmatch(branch)
    if match is None:
        raise ReleaseError(f"not a generated release branch: {branch}")
    return Release(package=match.group("package"), version=match.group("version"))


def read_package_version(root: Path, package: str) -> str:
    """Read a workspace package version without executing project code."""
    if package not in PACKAGES:
        raise ReleaseError(f"unsupported package: {package}")
    pyproject = root / "packages" / package / "pyproject.toml"
    with pyproject.open("rb") as file:
        version = tomllib.load(file).get("project", {}).get("version")
    if not isinstance(version, str) or re.fullmatch(_VERSION, version) is None:
        raise ReleaseError(f"invalid version in {pyproject}: {version!r}")
    return version


def validate_release_version(root: Path, release: Release) -> None:
    """Require branch metadata to match the reviewed package metadata."""
    actual = read_package_version(root, release.package)
    if actual != release.version:
        raise ReleaseError(f"release branch version {release.version} does not match package version {actual}")


def expected_release_files(release: Release) -> tuple[set[str], set[str]]:
    """Return required and allowed paths for a generated release PR."""
    package_root = release.package_dir.as_posix()
    required = {
        f"{package_root}/CHANGELOG.md",
        f"{package_root}/pyproject.toml",
    }
    allowed = required | {"requirements.txt", "uv.lock"}
    return required, allowed


def validate_changed_files(release: Release, changed_files: set[str]) -> None:
    """Fail closed when a release PR omits or adds a path outside its contract."""
    required, allowed = expected_release_files(release)
    missing = required - changed_files
    unexpected = changed_files - allowed
    if missing:
        raise ReleaseError(f"release is missing required files: {', '.join(sorted(missing))}")
    if unexpected:
        raise ReleaseError(f"release contains unexpected files: {', '.join(sorted(unexpected))}")


def _voicepad_dependencies(root: Path) -> list[str]:
    pyproject = root / "packages" / "voicepad" / "pyproject.toml"
    with pyproject.open("rb") as file:
        dependencies = tomllib.load(file).get("project", {}).get("dependencies")
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise ReleaseError(f"invalid dependencies in {pyproject}")
    return dependencies


def expected_core_dependency(root: Path) -> str:
    """Return the minimum core requirement for a release from this workspace."""
    return f"voicepad-core>={read_package_version(root, 'voicepad-core')}"


def validate_core_dependency(root: Path) -> str:
    """Require voicepad to depend on at least the exact local core release."""
    expected = expected_core_dependency(root)
    matches = [item for item in _voicepad_dependencies(root) if item.startswith("voicepad-core")]
    if matches != [expected]:
        raise ReleaseError(f"voicepad dependency must be exactly {expected!r}; found {matches!r}")
    return expected.removeprefix("voicepad-core>=")


def sync_core_dependency(root: Path) -> bool:
    """Update voicepad's core lower bound to the exact local core version."""
    path = root / "packages" / "voicepad" / "pyproject.toml"
    text = path.read_text()
    expected = expected_core_dependency(root)
    replacement = rf'\g<indent>"{expected}",\g<suffix>'
    updated, count = _CORE_DEPENDENCY.subn(replacement, text)
    if count != 1:
        raise ReleaseError(f"expected one voicepad-core dependency in {path}; found {count}")
    if updated == text:
        return False
    path.write_text(updated)
    return True


def artifact_digests(directory: Path) -> dict[str, str]:
    """Hash the isolated wheel and source distribution produced for one package."""
    if not directory.is_dir():
        raise ReleaseError(f"artifact directory does not exist: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.name != ".gitignore")
    if not files:
        raise ReleaseError(f"artifact directory is empty: {directory}")
    unexpected = [path.name for path in files if not (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))]
    if unexpected:
        raise ReleaseError(f"unexpected release artifacts: {', '.join(unexpected)}")
    wheels = [path for path in files if path.name.endswith(".whl")]
    source_distributions = [path for path in files if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(source_distributions) != 1:
        raise ReleaseError("release must contain exactly one wheel and one source distribution")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def validate_artifact_names(artifacts: set[str], package: str, version: str) -> None:
    """Require filenames for exactly the selected normalized package and version."""
    normalized = package.replace("-", "_")
    source = f"{normalized}-{version}.tar.gz"
    wheel_prefix = f"{normalized}-{version}-"
    wheels = {filename for filename in artifacts if filename.startswith(wheel_prefix) and filename.endswith(".whl")}
    if source not in artifacts or len(wheels) != 1 or artifacts != wheels | {source}:
        raise ReleaseError(f"artifacts do not match {package} {version}: {', '.join(sorted(artifacts))}")


def missing_pypi_artifacts(local: dict[str, str], remote: dict[str, str] | None) -> list[str]:
    """Return artifacts still publishable after validating any existing subset."""
    if remote is None:
        return sorted(local)
    conflicts = {filename for filename, digest in remote.items() if local.get(filename) != digest}
    if conflicts:
        raise ReleaseError(f"PyPI contains conflicting artifacts: {', '.join(sorted(conflicts))}")
    return sorted(set(local) - set(remote))


def compare_pypi_artifacts(local: dict[str, str], remote: dict[str, str] | None) -> str:
    """Classify absent, resumable partial, matching, or conflicting PyPI state."""
    try:
        missing = missing_pypi_artifacts(local, remote)
    except ReleaseError:
        return "conflict"
    if remote is None:
        return "absent"
    if missing:
        return "partial"
    return "matching"


def fetch_pypi_artifacts(package: str, version: str) -> dict[str, str] | None:
    """Read public PyPI artifact hashes for an exact package version."""
    if package not in PACKAGES or re.fullmatch(_VERSION, version) is None:
        raise ReleaseError("invalid package or version for PyPI lookup")
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "voicepad-release-workflow"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed PyPI origin
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise ReleaseError(f"PyPI lookup failed with HTTP {error.code}") from error
    except (OSError, ValueError) as error:
        raise ReleaseError(f"PyPI lookup failed: {error}") from error

    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise ReleaseError("PyPI response has no artifact list")
    artifacts: dict[str, str] = {}
    for item in urls:
        if not isinstance(item, dict):
            raise ReleaseError("PyPI response contains invalid artifact metadata")
        filename = item.get("filename")
        digests = item.get("digests")
        digest = digests.get("sha256") if isinstance(digests, dict) else None
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ReleaseError("PyPI response contains an artifact without a SHA-256 digest")
        if filename in artifacts:
            raise ReleaseError(f"PyPI response repeats artifact filename: {filename}")
        artifacts[filename] = digest
    if not artifacts:
        raise ReleaseError("PyPI version exists without downloadable artifacts")
    return artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-input")
    validate.add_argument("--package", required=True)
    validate.add_argument("--bump", required=True)

    version = subparsers.add_parser("version")
    version.add_argument("--package", required=True)

    sync = subparsers.add_parser("sync-core-dependency")
    sync.add_argument("--package", required=True)

    branch_field = subparsers.add_parser("branch-field")
    branch_field.add_argument("--branch", required=True)
    branch_field.add_argument("--field", choices=("package", "version", "tag", "prerelease"), required=True)

    validate_version = subparsers.add_parser("validate-version")
    validate_version.add_argument("--branch", required=True)

    validate_files = subparsers.add_parser("validate-files")
    validate_files.add_argument("--branch", required=True)
    validate_files.add_argument("--file", action="append", default=[])

    core = subparsers.add_parser("validate-core-dependency")
    core.add_argument("--require-published", action="store_true")

    for command in ("pypi-status", "pypi-files"):
        pypi = subparsers.add_parser(command)
        pypi.add_argument("--package", required=True)
        pypi.add_argument("--version", required=True)
        pypi.add_argument("--dist", type=Path, required=True)

    return parser


def _run(args: argparse.Namespace) -> None:
    root: Path = args.root.resolve()
    if args.command == "validate-input":
        validate_input(args.package, args.bump)
    elif args.command == "version":
        print(read_package_version(root, args.package))
    elif args.command == "sync-core-dependency":
        validate_input(args.package, "patch")
        if args.package == "voicepad":
            sync_core_dependency(root)
    elif args.command == "branch-field":
        release = parse_release_branch(args.branch)
        values = {
            "package": release.package,
            "version": release.version,
            "tag": release.tag,
            "prerelease": str(release.is_prerelease).lower(),
        }
        print(values[args.field])
    elif args.command == "validate-version":
        validate_release_version(root, parse_release_branch(args.branch))
    elif args.command == "validate-files":
        validate_changed_files(parse_release_branch(args.branch), set(args.file))
    elif args.command == "validate-core-dependency":
        version = validate_core_dependency(root)
        if args.require_published and fetch_pypi_artifacts("voicepad-core", version) is None:
            raise ReleaseError(f"voicepad-core {version} must be published before voicepad")
    elif args.command in {"pypi-status", "pypi-files"}:
        validate_input(args.package, "patch")
        local = artifact_digests(args.dist)
        validate_artifact_names(set(local), args.package, args.version)
        remote = fetch_pypi_artifacts(args.package, args.version)
        if args.command == "pypi-status":
            state = compare_pypi_artifacts(local, remote)
            if state == "conflict":
                raise ReleaseError("PyPI already contains different artifacts for this package version")
            print(state)
        else:
            for filename in missing_pypi_artifacts(local, remote):
                print(args.dist / filename)
    else:  # pragma: no cover - argparse enforces the command choices
        raise AssertionError(args.command)


def main() -> int:
    """Run the release validation CLI."""
    try:
        _run(_parser().parse_args())
    except ReleaseError as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
