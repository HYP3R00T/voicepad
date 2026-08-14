from __future__ import annotations

import os
import tempfile
from pathlib import Path

from voicepad_core.pipeline import TranscriptionResult


def render_markdown(audio_path: Path, result: TranscriptionResult) -> str:
    active = result.deployment
    warnings = "\n".join(f"  - {warning}" for warning in result.warnings) or "  []"
    return "\n".join((
        "---",
        f"audio: {audio_path.name}",
        f"deployment: {active.definition.id}",
        f"model: {active.definition.model_id}",
        f"artifact_revision: {active.snapshot_revision}",
        f"device_id: {active.device_id}",
        f"precision: {active.definition.precision}",
        f"duration_seconds: {result.duration_seconds:.3f}",
        f"latency_seconds: {result.latency_seconds:.3f}",
        f"complete: {'true' if result.complete else 'false'}",
        f"chunks: {len(result.chunks)}",
        "warnings:",
        warnings,
        "---",
        "",
        result.text,
        "",
    ))


def persist_markdown(audio_path: Path, result: TranscriptionResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{audio_path.stem}.md"
    descriptor, temporary_name = tempfile.mkstemp(dir=directory, prefix=f".{audio_path.stem}-", suffix=".md")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(render_markdown(audio_path, result))
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
