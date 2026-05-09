"""Microphone capture for push-to-talk dictation.

The entire public API is three methods:

    recorder.start()          — open the mic, begin collecting samples
    audio = recorder.stop()   — close the mic, return float32 array at 16 kHz
    recorder.is_recording()   — True while mic is open

That array goes straight into transcribe_buffer(). Nothing else happens here.

Sample rate handling:
    Whisper requires 16 kHz mono audio. Some devices (especially WASAPI on
    Windows and ALSA/PipeWire on Linux) only accept their native sample rate
    (commonly 44100 or 48000 Hz) and reject 16000 Hz with PaErrorCode -9997.

    AudioRecorder handles this transparently:
    - It queries the device's native sample rate before opening the stream.
    - If the native rate differs from 16000 Hz, it records at the native rate
      and resamples to 16000 Hz in stop() using scipy.signal.resample_poly.
    - The caller always receives a 16 kHz float32 array regardless of device.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soundfile as sf

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)

# Target sample rate for Whisper — never changes
SAMPLE_RATE: int = 16000
CHANNELS: int = 1


class AudioRecorderError(Exception):
    """Raised when the recorder cannot start or stop cleanly."""


def _query_device_sample_rate(device_index: int | None) -> int:
    """Return the native sample rate for a device index.

    Falls back to 16000 if the device cannot be queried.
    """
    try:
        info = sd.query_devices(device_index, kind="input")
        rate = int(info.get("default_samplerate", SAMPLE_RATE))
        return rate if rate > 0 else SAMPLE_RATE
    except Exception:
        return SAMPLE_RATE


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a float32 mono array from from_rate to to_rate.

    Uses scipy.signal.resample_poly for high-quality integer-ratio resampling.
    Falls back to linear interpolation if scipy is unavailable.
    """
    if from_rate == to_rate:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(to_rate, from_rate)
        up, down = to_rate // g, from_rate // g
        return resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        # scipy not available — fall back to linear interpolation
        logger.warning("scipy not available; using linear interpolation for resampling")
        n_out = int(len(audio) * to_rate / from_rate)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


class AudioRecorder:
    """Captures microphone audio as a numpy array at 16 kHz.

    Records at the device's native sample rate and resamples to 16 kHz on
    stop() if needed. This avoids PaErrorCode -9997 (Invalid sample rate)
    on WASAPI (Windows) and ALSA/PipeWire (Linux) devices that reject 16 kHz.

    Example:
        recorder = AudioRecorder(config)
        recorder.start()
        # ... user speaks ...
        audio = recorder.stop()                    # float32, 16 kHz, mono
        result = transcribe_buffer(audio, config)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._capture_rate: int = SAMPLE_RATE  # actual rate used for the open stream
        self._ensure_recordings_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone and start collecting audio.

        Queries the device's native sample rate first. If it differs from
        16 kHz, records at the native rate and resamples in stop().

        Raises:
            AudioRecorderError: If already recording or the device cannot be opened.
        """
        if self._recording:
            raise AudioRecorderError("Already recording")

        self._frames = []

        # Query native rate — avoids PaErrorCode -9997 on WASAPI/ALSA devices
        native_rate = _query_device_sample_rate(self.config.input_device_index)
        if native_rate != SAMPLE_RATE:
            logger.info(f"Device native rate is {native_rate} Hz — will resample to {SAMPLE_RATE} Hz after recording")
        self._capture_rate = native_rate

        try:
            self._stream = sd.InputStream(
                device=self.config.input_device_index,
                channels=CHANNELS,
                samplerate=self._capture_rate,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise AudioRecorderError(f"Cannot open audio device: {e}") from e

        self._recording = True
        logger.info(f"Recording started (device={self.config.input_device_index}, rate={self._capture_rate} Hz)")

    def stop(self) -> np.ndarray:
        """Close the microphone and return all captured audio at 16 kHz.

        Resamples from the device's native rate to 16 kHz if needed.

        Returns:
            float32 numpy array at 16 kHz mono.
            Empty array (length 0) if nothing was captured.

        Raises:
            AudioRecorderError: If not currently recording.
        """
        if not self._recording:
            raise AudioRecorderError("Not recording")

        self._recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
            finally:
                self._stream = None

        with self._lock:
            frames = list(self._frames)

        if not frames:
            logger.warning("No audio captured")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames).flatten()

        # Resample to 16 kHz if the device ran at a different rate
        if self._capture_rate != SAMPLE_RATE:
            audio = _resample(audio, self._capture_rate, SAMPLE_RATE)
            logger.debug(f"Resampled {self._capture_rate} Hz → {SAMPLE_RATE} Hz")

        logger.info(f"Recording stopped — {len(audio) / SAMPLE_RATE:.2f}s captured")
        return audio

    def is_recording(self) -> bool:
        """Return True while the microphone is open."""
        return self._recording

    def save_wav(self, audio: np.ndarray, path: Path) -> None:
        """Save a float32 audio array to a 16-bit WAV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        logger.debug(f"Saved {len(audio) / SAMPLE_RATE:.2f}s → {path}")

    def make_wav_path(self, prefix: str | None = None) -> Path:
        """Return a timestamped WAV path under the recordings directory."""
        from datetime import datetime

        pfx = prefix or self.config.recording_prefix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.recordings_path / f"{pfx}_{ts}.wav"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning(f"Stream status: {status}")
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())

    def _ensure_recordings_dir(self) -> None:
        path = self.config.recordings_path
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise AudioRecorderError(f"Cannot create recordings directory '{path}': {e}") from e
