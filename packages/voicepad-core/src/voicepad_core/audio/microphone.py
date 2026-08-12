from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import numpy as np
import sounddevice as sd

from .constants import DEFAULT_INPUT_CHANNELS, FALLBACK_INPUT_SAMPLE_RATE
from .errors import AudioStreamStateError
from .persistence import LiveWavRecording, WavArtifact
from .types import AudioWindow

logger = logging.getLogger(__name__)

CaptureFailureStage = Literal[
    "capture-write",
    "stream-finished",
    "native-stop",
    "native-close",
    "audio-finalization",
]


@dataclass(frozen=True, slots=True)
class CaptureFailure:
    """One privacy-safe stage marker retaining its original capture exception."""

    stage: CaptureFailureStage
    error: Exception

    @property
    def summary(self) -> str:
        return f"{self.stage}: {type(self.error).__name__}: {self.error}"


def _resolve_input_device(device_index: int | None) -> int | None:
    if sys.platform == "linux":
        if device_index is not None:
            logger.info(
                "Ignoring configured input device %s on Linux; using the shared system default",
                device_index,
            )
        return None
    return device_index


class MicrophoneStream:
    """Live microphone capture using a non-blocking sounddevice InputStream."""

    def __init__(self, recording_path: Path, device_index: int | None = None) -> None:
        self._device_index = _resolve_input_device(device_index)
        self._recording_path = recording_path
        self._sample_rate = FALLBACK_INPUT_SAMPLE_RATE
        self._channels = DEFAULT_INPUT_CHANNELS
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._live_recording: LiveWavRecording | None = None
        self._failures: list[CaptureFailure] = []
        self._last_artifact: WavArtifact | None = None
        self._started_at = 0.0
        self._recording = False
        self._logger = logging.getLogger(__name__)

    @property
    def sample_rate(self) -> int:
        """Return the input device's native sample rate."""
        return self._sample_rate

    @property
    def is_recording(self) -> bool:
        """Return whether capture still needs an explicit stop/finalization."""
        with self._lock:
            return self._recording

    @property
    def capture_failures(self) -> tuple[CaptureFailure, ...]:
        """Return every fatal capture/stop failure in occurrence order."""
        with self._lock:
            return tuple(self._failures)

    @property
    def capture_error(self) -> Exception | None:
        """Return the first fatal capture error for lightweight health polling."""
        failures = self.capture_failures
        return failures[0].error if failures else None

    @property
    def last_artifact(self) -> WavArtifact | None:
        """Return the artifact produced by the latest successful finalization."""
        with self._lock:
            return self._last_artifact

    @property
    def growing_source(self) -> LiveWavRecording:
        """Return the active disk-backed source for transcription workers."""
        with self._lock:
            if self._live_recording is None:
                raise AudioStreamStateError("MicrophoneStream has no active recording writer.")
            return self._live_recording

    def start(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._recording:
                    raise AudioStreamStateError("MicrophoneStream is already recording. Call stop() first.")
            live_recording = LiveWavRecording(self._recording_path, self._sample_rate, self._channels)
            live_recording.start()
            with self._lock:
                self._live_recording = live_recording
                self._failures.clear()
                self._last_artifact = None
                self._started_at = monotonic()
                self._recording = True

            native_stream: sd.InputStream | None = None
            try:
                native_stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="float32",
                    device=self._device_index,
                    callback=self._callback,
                    finished_callback=self._stream_finished,
                )
                with self._lock:
                    self._stream = native_stream
                native_stream.start()
            except Exception as error:
                with self._lock:
                    self._recording = False
                    self._stream = None
                    self._live_recording = None
                removed = live_recording.abort()
                if not removed:
                    self._logger.error(
                        "Recording startup failed and its spool remains recoverable: path=%s", self._recording_path
                    )
                if native_stream is not None:
                    try:
                        native_stream.close()
                    except Exception as close_error:
                        self._logger.warning("Failed to close microphone stream after start error: %s", close_error)
                if "PaErrorCode -9985" in str(error):
                    raise AudioStreamStateError(
                        "The shared system microphone is unavailable. Select a default input in Linux sound settings "
                        "and check its application permissions."
                    ) from error
                raise

        self._logger.info(
            "Microphone capture started: path=%s device=%s sample_rate=%s channels=%s",
            self._recording_path,
            self._device_index if self._device_index is not None else "system-default",
            self._sample_rate,
            self._channels,
        )

    def stop(self) -> WavArtifact:
        """Stop native capture and always attempt durable WAV finalization."""
        with self._lifecycle_lock:
            with self._lock:
                if self._last_artifact is not None:
                    return self._last_artifact
                if self._live_recording is None:
                    raise AudioStreamStateError("MicrophoneStream is not recording. Call start() first.")
                self._recording = False
                native_stream = self._stream
                self._stream = None
                live_recording = self._live_recording

            if native_stream is not None:
                self._run_native_cleanup("native-stop", native_stream.stop)
                self._run_native_cleanup("native-close", native_stream.close)

            try:
                artifact = live_recording.finish()
            except Exception as error:
                self._record_failure("audio-finalization", error)
                raise AudioStreamStateError(
                    f"Audio finalization failed; recoverable spool audio was retained for {self._recording_path}."
                ) from error

            with self._lock:
                self._last_artifact = artifact
            self._log_stopped(artifact)
            return artifact

    def _run_native_cleanup(
        self,
        stage: Literal["native-stop", "native-close"],
        operation: Callable[[], object],
    ) -> None:
        try:
            operation()
        except Exception as error:
            self._record_failure(stage, error)

    def _log_stopped(self, artifact: WavArtifact) -> None:
        elapsed = monotonic() - self._started_at
        self._logger.info(
            "Microphone capture stopped: path=%s elapsed_s=%.3f persisted_frames=%s "
            "persisted_duration_s=%.3f missing_duration_s=%.3f failures=%s",
            artifact.path,
            elapsed,
            artifact.frame_count,
            artifact.duration_s,
            max(0.0, elapsed - artifact.duration_s),
            len(self.capture_failures),
        )

    def read_window(self, start_sample: int, max_samples: int | None = None) -> AudioWindow:
        """Read committed capture beginning at an absolute sample position."""
        if self._live_recording is None:
            raise AudioStreamStateError("MicrophoneStream has no active recording writer.")
        return self._live_recording.read_from(start_sample, max_samples)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        callback_time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, callback_time
        if status:
            self._logger.warning("Microphone callback status: path=%s status=%s", self._recording_path, status)
        with self._lock:
            if not self._recording:
                return
            copied = indata.copy()
            live_recording = self._live_recording
            if live_recording is None:
                error = AudioStreamStateError("Microphone writer disappeared during active capture.")
                self._record_failure("capture-write", error)
                raise sd.CallbackAbort from error
            try:
                live_recording.append(copied)
            except Exception as error:
                self._record_failure("capture-write", error)
                raise sd.CallbackAbort from error

    def _stream_finished(self) -> None:
        with self._lock:
            if not self._recording or self._failures:
                return
            self._record_failure(
                "stream-finished",
                AudioStreamStateError("Microphone input stream stopped unexpectedly."),
            )

    def _record_failure(self, stage: CaptureFailureStage, error: Exception) -> None:
        failure = CaptureFailure(stage, error)
        with self._lock:
            self._failures.append(failure)
        self._logger.error(
            "Microphone capture failure: path=%s stage=%s error_type=%s error=%s",
            self._recording_path,
            stage,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
