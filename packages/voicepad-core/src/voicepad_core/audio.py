"""Microphone capture with automatic resampling to 16 kHz.

Records at device's native sample rate to avoid PaErrorCode -9997 on
WASAPI/ALSA devices that reject 16 kHz. Resamples to 16 kHz on stop().
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

SAMPLE_RATE: int = 16000  # Target sample rate for Whisper
CHANNELS: int = 1


class AudioRecorderError(Exception):
    """Raised when the recorder cannot start or stop cleanly."""


def _query_device_sample_rate(device_index: int | None) -> int:
    """Query device's native sample rate, fallback to 16000."""
    try:
        info = sd.query_devices(device_index, kind="input")
        rate = int(info.get("default_samplerate", SAMPLE_RATE))
        return rate if rate > 0 else SAMPLE_RATE
    except Exception:
        return SAMPLE_RATE


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio using scipy.signal.resample_poly, fallback to linear interpolation."""
    if from_rate == to_rate:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(to_rate, from_rate)
        up, down = to_rate // g, from_rate // g
        return resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        logger.warning("scipy not available; using linear interpolation for resampling")
        n_out = int(len(audio) * to_rate / from_rate)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


class AudioRecorder:
    """Captures microphone audio and resamples to 16 kHz.

    Records at device's native sample rate to avoid compatibility issues,
    then resamples to 16 kHz (required by Whisper) when recording stops.

    Attributes:
        config: Configuration with input device settings
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._capture_rate: int = SAMPLE_RATE
        self._ensure_recordings_dir()

    def start(self) -> None:
        """Open microphone and start collecting audio.

        Raises:
            AudioRecorderError: If already recording or device cannot be opened
        """
        if self._recording:
            raise AudioRecorderError("Already recording")

        self._frames = []

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
        """Close microphone and return captured audio at 16 kHz.

        Returns:
            float32 numpy array at 16 kHz mono (empty if nothing captured)

        Raises:
            AudioRecorderError: If not currently recording
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

        audio = np.concatenate(frames)
        if audio.ndim > 1:
            # Average stereo channels to mono (not interleave)
            audio = audio.mean(axis=1)
        audio = audio.flatten()

        if self._capture_rate != SAMPLE_RATE:
            audio = _resample(audio, self._capture_rate, SAMPLE_RATE)
            logger.debug(f"Resampled {self._capture_rate} Hz → {SAMPLE_RATE} Hz")

        logger.info(f"Recording stopped — {len(audio) / SAMPLE_RATE:.2f}s captured")
        return audio

    def is_recording(self) -> bool:
        """Check if microphone is currently recording."""
        return self._recording

    def save_wav(self, audio: np.ndarray, path: Path) -> None:
        """Save audio array to 16-bit WAV file.

        Args:
            audio: float32 audio array at 16 kHz
            path: Output file path (parent directories created automatically)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        logger.debug(f"Saved {len(audio) / SAMPLE_RATE:.2f}s → {path}")

    def make_wav_path(self, prefix: str | None = None) -> Path:
        """Generate timestamped WAV path in recordings directory.

        Args:
            prefix: Optional filename prefix (defaults to config.recording_prefix)

        Returns:
            Path with format: {prefix}_{YYYYMMDD_HHMMSS}.wav
        """
        from datetime import datetime

        pfx = prefix or self.config.recording_prefix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.recordings_path / f"{pfx}_{ts}.wav"

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
