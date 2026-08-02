from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from threading import RLock

from voicepad_core.deployments import WheelExtraction, get_manifest

from .store import COPY_BUFFER_SIZE, ArtifactIntegrityError, ArtifactStore

EXTRACTION_PROVENANCE_FILE = "voicepad-extraction.json"


class WheelExtractor:
    """Extract one declared data file from a verified wheel without importing it."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._root = store.root / "extractions"
        self._lock = RLock()

    def output_path(self, extraction: WheelExtraction) -> Path:
        return self._root / extraction.id / extraction.output_name

    def prepare(self, extraction: WheelExtraction) -> Path:
        with self._lock:
            target = self._root / extraction.id
            if os.path.lexists(target):
                return self.verify(extraction)

            manifest = get_manifest(extraction.wheel_manifest_id)
            snapshot = self._store.prepare(manifest)
            if len(manifest.files) != 1:
                raise ArtifactIntegrityError("Wheel extraction requires exactly one verified wheel artifact.")
            wheel = snapshot / manifest.files[0].path
            self._root.mkdir(parents=True, exist_ok=True)
            operation = Path(tempfile.mkdtemp(prefix=f".{extraction.id}-", dir=self._store.root))
            staged = operation / "extraction"
            staged.mkdir()
            try:
                self._extract(wheel, staged / extraction.output_name, extraction)
                _write_provenance(staged / EXTRACTION_PROVENANCE_FILE, extraction)
                os.replace(staged, target)
                return self.verify(extraction)
            finally:
                shutil.rmtree(operation, ignore_errors=True)

    def verify(self, extraction: WheelExtraction) -> Path:
        target = self._root / extraction.id
        model = target / extraction.output_name
        _verify_model(model, extraction)
        provenance = target / EXTRACTION_PROVENANCE_FILE
        try:
            recorded = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(f"Silero extraction provenance is invalid: {extraction.id}") from error
        if recorded != _provenance(extraction):
            raise ArtifactIntegrityError(f"Silero extraction provenance does not match: {extraction.id}")
        return model

    @staticmethod
    def _extract(wheel: Path, destination: Path, extraction: WheelExtraction) -> None:
        try:
            with zipfile.ZipFile(wheel) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if len(names) != len(set(names)):
                    raise ArtifactIntegrityError("Official Silero wheel contains duplicate entries.")
                for name in names:
                    path = PurePosixPath(name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ArtifactIntegrityError("Official Silero wheel contains an unsafe path.")
                matches = [entry for entry in entries if entry.filename == extraction.entry_path]
                if len(matches) != 1:
                    raise ArtifactIntegrityError("Official Silero ONNX entry is missing or duplicated.")
                entry = matches[0]
                if (
                    entry.is_dir()
                    or entry.file_size != extraction.size
                    or entry.compress_size != extraction.compressed_size
                ):
                    raise ArtifactIntegrityError("Official Silero ONNX entry metadata does not match.")
                with archive.open(entry) as source, destination.open("xb") as output:
                    digest = hashlib.sha256()
                    copied = 0
                    while chunk := source.read(COPY_BUFFER_SIZE):
                        copied += len(chunk)
                        if copied > extraction.size:
                            raise ArtifactIntegrityError("Extracted Silero model exceeds its declared size.")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if copied != extraction.size or digest.hexdigest() != extraction.sha256:
                    raise ArtifactIntegrityError("Extracted Silero model bytes do not match.")
        except zipfile.BadZipFile as error:
            raise ArtifactIntegrityError("Official Silero artifact is not a valid wheel.") from error


def _verify_model(path: Path, extraction: WheelExtraction) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != extraction.size:
        raise ArtifactIntegrityError("Extracted Silero model is missing or has the wrong size.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != extraction.sha256:
        raise ArtifactIntegrityError("Extracted Silero model SHA-256 does not match.")


def _provenance(extraction: WheelExtraction) -> dict[str, object]:
    return {
        "schema": 1,
        "extraction_id": extraction.id,
        "wheel_manifest_id": extraction.wheel_manifest_id,
        "entry_path": extraction.entry_path,
        "output_name": extraction.output_name,
        "size": extraction.size,
        "compressed_size": extraction.compressed_size,
        "sha256": extraction.sha256,
    }


def _write_provenance(path: Path, extraction: WheelExtraction) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(_provenance(extraction), file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
