from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

from .constants import DEFAULT_INPUT_CHANNELS, FALLBACK_INPUT_SAMPLE_RATE
from .errors import AudioStreamStateError
from .persistence import LiveWavRecording, WavArtifact
from .types import AudioWindow

logger = logging.getLogger(__name__)


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
        self._capture_error: Exception | None = None
        self._started_at = 0.0
        self._recording = False
        self._logger = logging.getLogger(__name__)

    @property
    def sample_rate(self) -> int:
        """Return the input device's native sample rate."""
        return self._sample_rate

    @property
    def is_recording(self) -> bool:
        """Return whether the stream is currently recording."""
        with self._lock:
            return self._recording

    @property
    def capture_error(self) -> Exception | None:
        """Return the first fatal capture or native-stream error, if any."""
        with self._lock:
            return self._capture_error

    @property
    def incremental_source(self) -> LiveWavRecording:
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
                self._capture_error = None
                self._started_at = time.monotonic()
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
                live_recording.abort()
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
            "Microphone capture started: path=%s device=%s sample_rate=%s",
            self._recording_path,
            self._device_index if self._device_index is not None else "system-default",
            self._sample_rate,
        )

    def stop(self) -> WavArtifact:
        """Stop capture and finalize its continuously written WAV file."""
        with self._lifecycle_lock:
            with self._lock:
                if not self._recording:
                    raise AudioStreamStateError("MicrophoneStream is not recording. Call start() first.")
                self._recording = False
                native_stream = self._stream
                self._stream = None
            if native_stream is not None:
                for operation in (native_stream.stop, native_stream.close):
                    try:
                        operation()
                    except Exception as error:
                        self._remember_error(error)
            if self._live_recording is None:
                raise AudioStreamStateError("MicrophoneStream has no active recording writer.")
            try:
                artifact = self._live_recording.finish()
            except Exception:
                self._logger.exception(
                    "Audio finalization failed; recoverable spool retained: %s", self._recording_path
                )
                raise
        elapsed = time.monotonic() - self._started_at
        self._logger.info(
            "Microphone capture stopped: path=%s elapsed_s=%.3f persisted_frames=%s "
            "persisted_duration_s=%.3f missing_duration_s=%.3f failed=%s",
            artifact.path,
            elapsed,
            artifact.frame_count,
            artifact.duration_s,
            max(0.0, elapsed - artifact.duration_s),
            self.capture_error is not None,
        )
        return artifact

    def read_window(self, start_sample: int, max_samples: int | None = None) -> AudioWindow:
        """Read committed audio beginning at an absolute source sample."""
        if self._live_recording is None:
            raise AudioStreamStateError("MicrophoneStream has no active recording writer.")
        return self._live_recording.read_from(start_sample, max_samples)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time
        if status:
            self._logger.warning("Microphone callback status: %s", status)
        with self._lock:
            if self._recording:
                copied = indata.copy()
                if self._live_recording is None:
                    raise sd.CallbackAbort
                try:
                    self._live_recording.append(copied)
                except Exception as error:
                    self._remember_error(error)
                    raise sd.CallbackAbort from error

    def _stream_finished(self) -> None:
        with self._lock:
            if self._recording and self._capture_error is None:
                self._capture_error = AudioStreamStateError("Microphone input stream stopped unexpectedly.")
                self._logger.error("Microphone input stream stopped unexpectedly: path=%s", self._recording_path)

    def _remember_error(self, error: Exception) -> None:
        with self._lock:
            if self._capture_error is None:
                self._capture_error = error
        self._logger.error(
            "Microphone capture failure: path=%s error_type=%s error=%s",
            self._recording_path,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
