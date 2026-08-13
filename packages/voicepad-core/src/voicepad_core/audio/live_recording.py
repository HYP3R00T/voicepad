"""Queue, persist, and expose audio while a recording is still growing."""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from utilityhub_logging import bind_context

from .errors import AudioStreamStateError, AudioWriteBackpressureError
from .types import AudioWindow
from .wav_persistence import WavArtifact, _finalize_live_wav, _read_wav_window


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
        logger: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
        log_context: Mapping[str, str] | None = None,
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
        self._committed_condition = threading.Condition(self._state_lock)
        self._thread: threading.Thread | None = None
        self._finishing = False
        self._spool_path: Path | None = None
        self._frame_count = 0
        self._error: Exception | None = None
        self._artifact: WavArtifact | None = None
        self._logger = logger or logging.getLogger(__name__)
        self._log_context = dict(log_context or {})

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def committed_samples(self) -> int:
        with self._state_lock:
            return self._frame_count

    @property
    def is_final(self) -> bool:
        with self._state_lock:
            return self._artifact is not None

    def read_range(self, start_sample: int, end_sample: int) -> AudioWindow:
        if end_sample <= start_sample:
            raise ValueError("read range end must be greater than its start")
        return self.read_from(start_sample, end_sample - start_sample)

    def wait_for_update(self, after_sample: int, timeout: float | None = None) -> tuple[int, bool]:
        with self._committed_condition:
            self._committed_condition.wait_for(
                lambda: self._frame_count > after_sample or self._artifact is not None or self._error is not None,
                timeout=timeout,
            )
            self._raise_if_failed()
            return self._frame_count, self._artifact is not None

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
        self._thread = threading.Thread(target=self._write_loop_with_context, name="audio-writer", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)
        self._raise_if_failed()
        if not self._ready.is_set():
            raise AudioStreamStateError("Timed out while starting the live audio writer.")
        self._logger.info(
            "Live audio writer started: destination=%s sample_rate=%s channels=%s queue_capacity=%s",
            self._path,
            self._sample_rate,
            self._channels,
            self._queue.maxsize,
        )

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
            artifact = _finalize_live_wav(
                self._spool_path,
                self._path,
                self._sample_rate,
                self._channels,
                self._frame_count,
            )
            with self._committed_condition:
                self._artifact = artifact
                self._thread = None
                self._committed_condition.notify_all()
            self._logger.info(
                "Live audio writer finalized: destination=%s frames=%s duration_s=%.3f",
                artifact.path,
                artifact.frame_count,
                artifact.duration_s,
            )
            return artifact
        finally:
            self._finished.set()

    def abort(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            with contextlib.suppress(queue.Full):
                self._queue.put(_FINISH, timeout=5.0)
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self._logger.error("Audio writer did not stop; retaining recoverable spool: %s", self._spool_path)
                return
        if self._spool_path is not None:
            self._spool_path.unlink(missing_ok=True)
        self._thread = None

    def _write_loop_with_context(self) -> None:
        with bind_context(**self._log_context):
            self._write_loop()

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
                            try:
                                spool.flush()
                                requested_end = (
                                    self._frame_count
                                    if item.max_samples is None
                                    else min(self._frame_count, item.start_sample + item.max_samples)
                                )
                                item.result = _read_wav_window(self._spool_path, item.start_sample, requested_end)
                            except Exception as error:
                                item.error = error
                                self._logger.exception(
                                    "Live transcription read failed without stopping audio persistence: "
                                    "destination=%s start_sample=%s max_samples=%s committed_frames=%s",
                                    self._path,
                                    item.start_sample,
                                    item.max_samples,
                                    self._frame_count,
                                )
                            finally:
                                item.completed.set()
                            continue
                        if isinstance(item, np.ndarray):
                            spool.write(cast("NDArray[np.float32]", item))
                            with self._committed_condition:
                                self._frame_count += len(item)
                                self._committed_condition.notify_all()
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self._logger.error(
                "Live audio writer failed: destination=%s error_type=%s error=%s",
                self._path,
                type(exc).__name__,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            with self._committed_condition:
                self._error = exc
                self._committed_condition.notify_all()
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
        return _read_wav_window(artifact.path, start_sample, end_sample)


__all__ = ["LiveWavRecording"]
