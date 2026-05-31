# audio/preprocessor.py

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .base import AudioSource

TARGET_SAMPLE_RATE = 16000  # Whisper requires exactly this


class AudioPreProcessor:
    """
    Normalizes audio from any AudioSource into what Whisper expects:
      - Sample rate: 16000 Hz
      - Channels:    Mono (1 channel)
      - Dtype:       float32
      - Range:       Normalized to [-1.0, 1.0]

    This is the single place where all audio normalization happens.
    No other component should perform resampling or channel conversion.
    """

    def __init__(self, source: AudioSource) -> None:
        """
        Args:
            source: Any AudioSource (MicrophoneSource or FileSource).
        """
        self._source = source

    def process(self) -> np.ndarray:
        """
        Read from the source and return clean, Whisper-ready audio.

        Returns:
            np.ndarray: float32 array, mono, 16kHz, normalized.
        """
        audio = self._source.read()
        sample_rate = self._source.get_sample_rate()
        channels = self._source.get_channels()

        print(f"[PreProcessor] Input: {len(audio)} samples, {sample_rate}Hz, {channels}ch")

        # Step 1: Convert to float32 (in case source gave int16 etc.)
        audio = self._to_float32(audio)

        # Step 2: Stereo → Mono
        audio = self._to_mono(audio, channels)

        # Step 3: Resample to 16kHz
        audio = self._resample(audio, sample_rate, TARGET_SAMPLE_RATE)

        # Step 4: Normalize amplitude to [-1.0, 1.0]
        audio = self._normalize(audio)

        print(f"[PreProcessor] Output: {len(audio)} samples, {TARGET_SAMPLE_RATE}Hz, mono, float32")

        return audio

    # ------------------------------------------------------------------
    # Private helpers — each step is isolated and testable
    # ------------------------------------------------------------------

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        """Ensure the array is float32."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio

    def _to_mono(self, audio: np.ndarray, channels: int) -> np.ndarray:
        """
        Convert stereo or multi-channel audio to mono by averaging channels.

        Averaging is correct — it preserves loudness across both channels.
        Interleaving (wrong approach) would corrupt the signal.
        """
        if channels == 1:
            # Already mono — just ensure 1D shape
            return audio.flatten()

        # Shape is (N, C) — average across channel axis
        # Shape is (N*C,) interleaved — reshape then average
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio.reshape(-1, channels).mean(axis=1)

        return audio.astype(np.float32)

    def _resample(
        self,
        audio: np.ndarray,
        from_rate: int,
        to_rate: int,
    ) -> np.ndarray:
        """
        Resample audio from from_rate to to_rate using polyphase filtering.

        resample_poly is used because:
        - It preserves audio quality better than linear interpolation
        - It handles non-integer ratios cleanly (e.g. 44100 → 16000)
        - It is significantly better than np.interp for audio
        """
        if from_rate == to_rate:
            return audio  # Nothing to do

        # Reduce the ratio to lowest terms for resample_poly
        common = gcd(from_rate, to_rate)
        up = to_rate // common
        down = from_rate // common

        print(f"[PreProcessor] Resampling {from_rate}Hz → {to_rate}Hz (ratio {up}/{down})")

        resampled = resample_poly(audio, up, down)
        return resampled.astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize amplitude so the peak is at 1.0.

        Skips normalization if audio is silent (all zeros)
        to avoid division by zero.
        """
        peak = np.max(np.abs(audio))
        if peak > 0.0:
            audio = audio / peak
        return audio
