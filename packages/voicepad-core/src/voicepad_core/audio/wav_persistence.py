"""Create and finalize durable WAV files without exposing partial output."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import PCM_WAV_SUBTYPE
from .errors import AudioStreamStateError
from .types import AudioWindow, RawAudio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WavArtifact:
    """Metadata for an atomically persisted WAV recording."""

    path: Path
    sample_rate: int
    channels: int
    frame_count: int
    duration_s: float

    def duration(self) -> float:
        """Return the persisted recording duration in seconds."""
        return self.duration_s


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
        _promote_new_file(temporary_path, destination)
        _flush_file(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    artifact = WavArtifact(
        path=destination,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        frame_count=audio.frame_count,
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


def _read_wav_window(path: Path, start_sample: int, end_sample: int | None = None) -> AudioWindow:
    with sf.SoundFile(str(path), mode="r") as recording:
        available = len(recording) if end_sample is None else end_sample
        start = min(start_sample, available)
        recording.seek(start)
        samples = recording.read(available - start, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        if samples.shape[1] != 1:
            raise AudioStreamStateError("Live transcription currently requires mono capture.")
        samples = samples[:, 0]
    return AudioWindow(np.ascontiguousarray(samples, dtype=np.float32), start)


def _finalize_live_wav(
    spool_path: Path,
    destination: Path,
    sample_rate: int,
    channels: int,
    frame_count: int,
) -> WavArtifact:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".wav",
    )
    os.close(descriptor)
    temporary_path = Path(name)
    try:
        with (
            sf.SoundFile(str(spool_path), mode="r") as source,
            sf.SoundFile(
                str(temporary_path),
                mode="w",
                samplerate=sample_rate,
                channels=channels,
                subtype=PCM_WAV_SUBTYPE,
                format="WAV",
            ) as output,
        ):
            while True:
                block = source.read(65_536, dtype="float32", always_2d=True)
                if len(block) == 0:
                    break
                output.write(block)
        _flush_file(temporary_path)
        _promote_new_file(temporary_path, destination)
        _flush_file(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    spool_path.unlink(missing_ok=True)

    artifact = WavArtifact(
        path=destination,
        sample_rate=sample_rate,
        channels=channels,
        frame_count=frame_count,
        duration_s=frame_count / sample_rate,
    )
    logger.info(
        "Finalized live WAV: path=%s, duration_s=%.3f, frames=%s, rate=%s, channels=%s",
        artifact.path,
        artifact.duration_s,
        artifact.frame_count,
        artifact.sample_rate,
        artifact.channels,
    )
    return artifact


def _promote_new_file(temporary_path: Path, destination: Path) -> None:
    os.link(temporary_path, destination)
    temporary_path.unlink()
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _flush_file(path: Path) -> None:
    with path.open("r+b") as persisted_file:
        os.fsync(persisted_file.fileno())


__all__ = ["WavArtifact", "write_wav_atomic"]
