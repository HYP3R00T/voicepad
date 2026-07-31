from __future__ import annotations

import contextlib
import logging
import os
import queue
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .constants import PCM_WAV_SUBTYPE
from .errors import AudioStreamStateError, AudioWriteBackpressureError
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


@dataclass
class _ReadRequest:
    start_sample: int
    max_samples: int | None
    completed: threading.Event
    result: AudioWindow | None = None
    error: Exception | None = None


_FINISH = object()


class LiveWavRecording:
    """Append live audio on a worker thread and expose sample-addressed windows."""

    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        channels: int,
        *,
        max_pending_chunks: int = 256,
    ) -> None:
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        if max_pending_chunks <= 0:
            raise ValueError("max_pending_chunks must be positive")
        self._path = Path(path)
        self._sample_rate = sample_rate
        self._channels = channels
        self._queue: queue.Queue[np.ndarray | _ReadRequest | object] = queue.Queue(max_pending_chunks)
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._finishing = False
        self._spool_path: Path | None = None
        self._frame_count = 0
        self._error: Exception | None = None
        self._artifact: WavArtifact | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise AudioStreamStateError("Live recording is already started.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.stem}-live-",
            suffix=".wav",
        )
        os.close(descriptor)
        self._spool_path = Path(name)
        self._thread = threading.Thread(target=self._write_loop, name="audio-writer", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)
        self._raise_if_failed()
        if not self._ready.is_set():
            raise AudioStreamStateError("Timed out while starting the live audio writer.")

    def append(self, samples: np.ndarray) -> None:
        self._require_active()
        self._raise_if_failed()
        try:
            self._queue.put_nowait(np.ascontiguousarray(samples, dtype=np.float32))
        except queue.Full as exc:
            raise AudioWriteBackpressureError(
                f"Audio writer exceeded its {self._queue.maxsize}-chunk backlog."
            ) from exc

    def read_from(self, start_sample: int, max_samples: int | None = None) -> AudioWindow:
        if start_sample < 0:
            raise ValueError("start_sample must not be negative")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")

        with self._state_lock:
            artifact = self._artifact
            finishing = self._finishing
            if artifact is None and not finishing:
                self._require_active()
                request = _ReadRequest(start_sample, max_samples, threading.Event())
                self._queue.put(request)
            else:
                request = None

        if artifact is not None:
            return self._read_artifact(artifact, start_sample, max_samples)
        if finishing:
            if not self._finished.wait(timeout=60.0):
                raise AudioStreamStateError("Timed out while waiting for live audio to finish.")
            self._raise_if_failed()
            if self._artifact is None:
                raise AudioStreamStateError("Live recording did not finish successfully.")
            return self._read_artifact(self._artifact, start_sample, max_samples)

        assert request is not None
        if not request.completed.wait(timeout=10.0):
            raise AudioStreamStateError("Timed out while reading live audio.")
        if request.error is not None:
            raise AudioStreamStateError("Could not read live audio.") from request.error
        assert request.result is not None
        return request.result

    def finish(self) -> WavArtifact:
        with self._state_lock:
            if self._artifact is not None:
                return self._artifact
            self._require_active()
            self._finishing = True
            self._queue.put(_FINISH)
            thread = self._thread

        assert thread is not None
        try:
            thread.join(timeout=60.0)
            if thread.is_alive():
                raise AudioStreamStateError("Timed out while finishing the live audio writer.")
            self._raise_if_failed()
            assert self._spool_path is not None
            artifact = _finalize_spool(
                self._spool_path,
                self._path,
                self._sample_rate,
                self._channels,
                self._frame_count,
            )
            with self._state_lock:
                self._artifact = artifact
                self._thread = None
            return artifact
        finally:
            self._finished.set()

    def abort(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            with contextlib.suppress(queue.Full):
                self._queue.put(_FINISH, timeout=5.0)
            self._thread.join(timeout=5.0)
        if self._spool_path is not None:
            self._spool_path.unlink(missing_ok=True)
        self._thread = None

    def _write_loop(self) -> None:
        assert self._spool_path is not None
        try:
            with sf.SoundFile(
                str(self._spool_path),
                mode="w",
                samplerate=self._sample_rate,
                channels=self._channels,
                subtype="FLOAT",
                format="WAV",
            ) as spool:
                self._ready.set()
                while True:
                    item = self._queue.get()
                    try:
                        if item is _FINISH:
                            break
                        if isinstance(item, _ReadRequest):
                            spool.flush()
                            requested_end = (
                                self._frame_count
                                if item.max_samples is None
                                else min(self._frame_count, item.start_sample + item.max_samples)
                            )
                            item.result = _read_window(self._spool_path, item.start_sample, requested_end)
                            item.completed.set()
                            continue
                        if isinstance(item, np.ndarray):
                            spool.write(cast("NDArray[np.float32]", item))
                            self._frame_count += len(item)
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self._error = exc
            self._ready.set()
            self._fail_pending_reads(exc)

    def _fail_pending_reads(self, error: Exception) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _ReadRequest):
                item.error = error
                item.completed.set()
            self._queue.task_done()

    def _require_active(self) -> None:
        if self._thread is None:
            raise AudioStreamStateError("Live recording is not active.")

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise AudioStreamStateError("Live audio writer failed.") from self._error

    @staticmethod
    def _read_artifact(artifact: WavArtifact, start_sample: int, max_samples: int | None) -> AudioWindow:
        end_sample = None if max_samples is None else start_sample + max_samples
        return _read_window(artifact.path, start_sample, end_sample)


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


def _read_window(path: Path, start_sample: int, end_sample: int | None = None) -> AudioWindow:
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


def _finalize_spool(
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
        os.replace(temporary_path, destination)
        _flush_file(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
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


def _flush_file(path: Path) -> None:
    with path.open("r+b") as persisted_file:
        os.fsync(persisted_file.fileno())


__all__ = ["LiveWavRecording", "WavArtifact", "write_wav_atomic"]
