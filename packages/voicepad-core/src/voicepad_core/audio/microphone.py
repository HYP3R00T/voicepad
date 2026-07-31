from __future__ import annotations

import logging
import sys
import threading
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


def _query_native_rate(device_index: int | None) -> int:
    try:
        info = sd.query_devices(device_index, kind="input")
        rate = int(info.get("default_samplerate", FALLBACK_INPUT_SAMPLE_RATE))
        return rate if rate > 0 else FALLBACK_INPUT_SAMPLE_RATE
    except Exception as err:
        logger.warning("Falling back to %sHz for input device %s: %s", FALLBACK_INPUT_SAMPLE_RATE, device_index, err)
        return FALLBACK_INPUT_SAMPLE_RATE


class MicrophoneStream:
    """Live microphone capture using a non-blocking sounddevice InputStream."""

    def __init__(self, recording_path: Path, device_index: int | None = None) -> None:
        self._device_index = _resolve_input_device(device_index)
        self._recording_path = recording_path
        self._sample_rate = _query_native_rate(self._device_index)
        self._channels = DEFAULT_INPUT_CHANNELS
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._live_recording: LiveWavRecording | None = None
        self._capture_error: Exception | None = None
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
                self._recording = True

            native_stream: sd.InputStream | None = None
            try:
                native_stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="float32",
                    device=self._device_index,
                    callback=self._callback,
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
            "MicrophoneStream started: device=%s, sample_rate=%s",
            self._device_index if self._device_index is not None else "system-default",
            self._sample_rate,
        )

    def stop(self) -> WavArtifact:
        """Stop capture and finalize its continuously written WAV file."""
        self._stop_native_stream()
        if self._live_recording is None:
            raise AudioStreamStateError("MicrophoneStream has no active recording writer.")
        artifact = self._live_recording.finish()
        self._raise_capture_error(artifact)
        self._log_stopped(artifact.frame_count)
        return artifact

    def _stop_native_stream(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if not self._recording:
                    raise AudioStreamStateError("MicrophoneStream is not recording. Call start() first.")
                self._recording = False
                native_stream = self._stream
                self._stream = None

            if native_stream is not None:
                try:
                    native_stream.stop()
                finally:
                    native_stream.close()

    def _log_stopped(self, frame_count: int) -> None:
        self._logger.info(
            "MicrophoneStream stopped: samples=%s, duration_s=%.3f",
            frame_count,
            frame_count / self._sample_rate,
        )

    def _raise_capture_error(self, artifact: WavArtifact) -> None:
        if self._capture_error is not None:
            raise AudioStreamStateError(f"Audio capture failed; partial recording was saved to {artifact.path}.") from (
                self._capture_error
            )

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
                    self._capture_error = error
                    raise sd.CallbackAbort from error
