from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .constants import PCM_WAV_SUBTYPE
from .types import RawAudio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WavArtifact:
    """Metadata for an atomically persisted WAV recording."""

    path: Path
    sample_rate: int
    channels: int
    frame_count: int
    duration_s: float


def write_wav_atomic(audio: RawAudio, path: str | Path) -> WavArtifact:
    """Persist raw audio as PCM WAV without exposing a partial destination file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".wav",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        sf.write(
            str(temporary_path),
            audio.samples,
            audio.sample_rate,
            subtype=PCM_WAV_SUBTYPE,
            format="WAV",
        )
        _flush_file(temporary_path)
        os.replace(temporary_path, destination)
        _flush_file(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    artifact = WavArtifact(
        path=destination,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        frame_count=audio.samples.shape[0],
        duration_s=audio.duration(),
    )
    logger.info(
        "Persisted WAV atomically: path=%s, duration_s=%.3f, frames=%s, rate=%s, channels=%s",
        artifact.path,
        artifact.duration_s,
        artifact.frame_count,
        artifact.sample_rate,
        artifact.channels,
    )
    return artifact


def _flush_file(path: Path) -> None:
    with path.open("r+b") as persisted_file:
        os.fsync(persisted_file.fileno())


__all__ = ["WavArtifact", "write_wav_atomic"]
