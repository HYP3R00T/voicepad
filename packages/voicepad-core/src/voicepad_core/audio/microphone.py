from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

from .constants import DEFAULT_INPUT_CHANNELS, DEFAULT_INPUT_SAMPLE_RATE, PCM_WAV_SUBTYPE
from .errors import AudioStreamStateError

logger = logging.getLogger(__name__)


def _query_native_rate(device_index: int | None) -> int:
    try:
        info = sd.query_devices(device_index, kind="input")
        rate = int(info.get("default_samplerate", DEFAULT_INPUT_SAMPLE_RATE))
        return rate if rate > 0 else DEFAULT_INPUT_SAMPLE_RATE
    except Exception as err:
        logger.warning("Falling back to %sHz for input device %s: %s", DEFAULT_INPUT_SAMPLE_RATE, device_index, err)
        return DEFAULT_INPUT_SAMPLE_RATE


class MicrophoneStream:
    """
    Live microphone capture using a non-blocking sounddevice InputStream.

    Frames are buffered during recording, exposed as snapshots for streaming
    consumers, and returned as one raw array when recording stops.
    """

    def __init__(self, device_index: int | None = None) -> None:
        """
        Args:
            device_index: Microphone device index. None uses the system default.
        """
        self._device_index = device_index
        self._sample_rate = _query_native_rate(device_index)
        self._channels = DEFAULT_INPUT_CHANNELS
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._logger = logging.getLogger(__name__)

    @property
    def sample_rate(self) -> int:
        """Return the input device's native sample rate."""
        return self._sample_rate

    @property
    def is_recording(self) -> bool:
        """Return whether the stream is currently recording."""
        return self._recording

    def start(self) -> None:
        """
        Open the microphone stream and begin buffering frames.

        Raises:
            AudioStreamStateError: If the stream is already recording.
        """
        if self._recording:
            raise AudioStreamStateError("MicrophoneStream is already recording. Call stop() first.")

        self._frames.clear()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            device=self._device_index,
            callback=self._callback,
        )
        self._stream.start()
        self._recording = True
        self._logger.info(
            "MicrophoneStream started: device_index=%s, sample_rate=%s",
            self._device_index,
            self._sample_rate,
        )

    def stop(self) -> np.ndarray:
        """
        Stop recording and return all accumulated raw audio.

        Raises:
            AudioStreamStateError: If the stream is not currently recording.
        """
        if not self._recording:
            raise AudioStreamStateError("MicrophoneStream is not recording. Call start() first.")

        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            out = np.zeros(0, dtype=np.float32) if not self._frames else np.concatenate(self._frames, axis=0).ravel()

        self._logger.info(
            "MicrophoneStream stopped: samples=%s, duration_s=%.3f",
            len(out),
            len(out) / self._sample_rate,
        )
        return out.astype(np.float32, copy=False)

    def get_snapshot(self) -> np.ndarray:
        """Return a thread-safe copy of the buffered raw audio."""
        with self._lock:
            out = (
                np.zeros(0, dtype=np.float32)
                if not self._frames
                else np.concatenate(self._frames, axis=0).ravel().copy()
            )
        self._logger.debug("MicrophoneStream snapshot: samples=%s", len(out))
        return out

    def save_wav(self, audio: np.ndarray, path: Path, sample_rate: int | None = None) -> None:
        """Write audio to disk as a 16-bit PCM WAV file."""
        rate = self._sample_rate if sample_rate is None else sample_rate
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, rate, subtype=PCM_WAV_SUBTYPE)
        self._logger.info("Saved WAV: %s (%.3fs, %s samples, rate=%s)", path, len(audio) / rate, len(audio), rate)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time, status
        if self._recording:
            self._frames.append(indata.copy())
