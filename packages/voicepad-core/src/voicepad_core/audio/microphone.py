# audio/microphone.py

from __future__ import annotations

import numpy as np
import sounddevice as sd

from .base import AudioSource


class MicrophoneSource(AudioSource):
    """
    Captures audio from the system microphone.

    Records at the device's native sample rate to avoid
    hardware rejection errors (common on WASAPI/ALSA devices
    that refuse to open at 16kHz directly).

    The PreProcessor will resample to 16kHz afterwards.
    """

    def __init__(self, device_index: int | None = None, duration_s: float = 5.0) -> None:
        """
        Args:
            device_index: Microphone device index. None = system default.
            duration_s:   How many seconds to record per read() call.
        """
        self._device_index = device_index
        self._duration_s = duration_s
        self._sample_rate = self._query_native_rate()
        self._channels = 1  # Always request mono from mic

    def _query_native_rate(self) -> int:
        """Ask sounddevice what sample rate the device natively supports."""
        try:
            info = sd.query_devices(self._device_index, kind="input")
            rate = int(info.get("default_samplerate", 16000))
            return rate if rate > 0 else 16000
        except Exception:
            # If query fails, fall back to 16000 safely
            return 16000

    def read(self) -> np.ndarray:
        """
        Block and record audio from the microphone.

        Returns:
            np.ndarray: float32 array of shape (N,) — flat mono audio.
        """
        print(f"[MicrophoneSource] Recording for {self._duration_s}s at {self._sample_rate}Hz...")

        audio = sd.rec(
            frames=int(self._duration_s * self._sample_rate),
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            device=self._device_index,
        )
        sd.wait()  # Block until recording is complete

        # Flatten from (N, 1) shape to (N,) flat array
        audio = audio.flatten()

        print(f"[MicrophoneSource] Captured {len(audio)} samples.")
        return audio

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_channels(self) -> int:
        return self._channels
