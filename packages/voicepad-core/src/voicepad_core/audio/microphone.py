# audio/microphone_stream.py

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


class MicrophoneStream:
    """
    Live microphone capture using a non-blocking sounddevice InputStream.

    Frames are accumulated in an internal buffer via a callback.
    The StreamingTranscriber calls get_snapshot() whenever it needs
    a copy of the current buffer to check for chunk boundaries.

    On stop(), all accumulated frames are concatenated and returned
    as a single float32 array at the device's native sample rate.
    The caller is responsible for resampling via AudioPreProcessor.

    WAV saving is also handled here because this is the only component
    that owns the complete raw audio for a session.
    """

    def __init__(self, device_index: int | None = None) -> None:
        """
        Args:
            device_index: Microphone device index. None = system default.
                          Run 'python -m sounddevice' to list devices.
        """
        self._device_index = device_index
        self._sample_rate: int = self._query_native_rate()
        self._channels: int = 1

        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Native sample rate of the input device."""
        return self._sample_rate

    @property
    def is_recording(self) -> bool:
        """True while the stream is open and accumulating frames."""
        return self._recording

    def start(self) -> None:
        """
        Open the microphone stream and begin accumulating frames.

        The stream runs in a background thread managed by sounddevice.
        This method returns immediately — recording happens in the background.

        Raises:
            RuntimeError: If the stream is already open.
        """
        if self._recording:
            raise RuntimeError("MicrophoneStream is already recording. Call stop() first.")

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
            f"MicrophoneStream started: device_index={self._device_index}, sample_rate={self._sample_rate}"
        )

    def stop(self) -> np.ndarray:
        """
        Stop recording and return all accumulated audio.

        Returns:
            np.ndarray: float32 array of shape (N,) at the device's
                        native sample rate. Mono.

        Raises:
            RuntimeError: If the stream is not currently recording.
        """
        if not self._recording:
            raise RuntimeError("MicrophoneStream is not recording. Call start() first.")

        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).flatten()

        out = audio.astype(np.float32)
        self._logger.info(
            f"MicrophoneStream stopped: samples={len(out)}, duration_s={len(out) / self._sample_rate:.3f}"
        )
        return out

    def get_snapshot(self) -> np.ndarray:
        """
        Return a thread-safe copy of all frames accumulated so far.

        Called by StreamingTranscriber during recording to check
        whether a chunk boundary has been reached. Does not consume
        or clear the buffer — recording continues unaffected.

        Returns:
            np.ndarray: float32 array of shape (N,) at native sample rate.
                        Returns empty array if nothing recorded yet.
        """
        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            out = np.concatenate(self._frames, axis=0).flatten().copy()
            # Lightweight debug log for snapshot retrieval
            self._logger.debug(f"MicrophoneStream snapshot: samples={len(out)}")
            return out

    def save_wav(
        self,
        audio: np.ndarray,
        path: Path,
        sample_rate: int | None = None,
    ) -> None:
        """
        Write audio to disk as a 16-bit PCM WAV file.

        This is intentionally a simple, synchronous write.
        The caller (StreamingTranscriber) is responsible for running
        this in a thread if parallel saving is required.

        Args:
            audio:       float32 mono array to save.
            path:        Full destination path including filename.
            sample_rate: Sample rate to embed in the WAV header.
                         Defaults to the device's native rate.

        Raises:
            OSError: If the directory cannot be created or file cannot be written.
        """
        rate = sample_rate if sample_rate is not None else self._sample_rate
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, rate, subtype="PCM_16")
        self._logger.info(f"Saved WAV: {path} ({len(audio) / rate:.3f}s, {len(audio)} samples, rate={rate})")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        """
        sounddevice calls this on every audio block (~20ms).

        Appends a copy of the incoming frame to _frames.
        The copy is critical — sounddevice reuses the indata buffer.

        This runs in sounddevice's internal audio thread.
        Only list.append() is used here — it is GIL-safe for
        single appends and does not require the lock.
        The lock is only needed in get_snapshot() and stop()
        where we iterate or concatenate.
        """
        if self._recording:
            self._frames.append(indata.copy())

    def _query_native_rate(self) -> int:
        """
        Query the device's native sample rate.

        Falls back to 16000 if the query fails — downstream
        resampling will handle the mismatch safely.
        """
        try:
            info = sd.query_devices(self._device_index, kind="input")
            rate = int(info.get("default_samplerate", 16000))
            return rate if rate > 0 else 16000
        except Exception:
            return 16000


# Backwards-compatible alias: older code/tests refer to MicrophoneSource
class MicrophoneSource:
    """
    Backwards-compatible blocking microphone interface expected by older
    tests: records for a fixed duration using `sd.rec` and `sd.wait`.
    """

    def __init__(self, device_index: int | None = None, duration_s: float = 5.0) -> None:
        self._device_index = device_index
        self._duration_s = duration_s
        self._sample_rate: int = self._query_native_rate()
        self._channels: int = 1

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_channels(self) -> int:
        return self._channels

    def read(self) -> np.ndarray:
        frames = int(self._duration_s * self._sample_rate)
        audio = sd.rec(
            frames=frames,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            device=self._device_index,
        )
        sd.wait()

        # Flatten mono (N,1) -> (N,)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.flatten()

        return audio.astype(np.float32)

    def _query_native_rate(self) -> int:
        """
        Query the device's native sample rate with the same fallback
        semantics as MicrophoneStream._query_native_rate.
        """
        try:
            info = sd.query_devices(self._device_index, kind="input")
            rate = int(info.get("default_samplerate", 16000))
            return rate if rate > 0 else 16000
        except Exception:
            return 16000
