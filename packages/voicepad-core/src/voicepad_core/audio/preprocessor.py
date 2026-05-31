# audio/preprocessor.py

from __future__ import annotations

from math import gcd

import numpy as np

from .base import AudioSource

TARGET_SAMPLE_RATE = 16_000  # Whisper requires exactly 16kHz


class AudioPreProcessor:
    """
    Normalizes audio from any AudioSource into what Whisper expects:

      - Sample rate : 16000 Hz
      - Channels    : Mono (1 channel)
      - Dtype       : float32
      - Amplitude   : Normalized to [-1.0, 1.0]

    This is the single place in the codebase where resampling
    and channel conversion happen. No other component should
    perform either of these operations.

    Resampling uses polyphase filtering (resample_poly) when scipy
    is available, falling back to linear interpolation (np.interp)
    if scipy is not installed. resample_poly is strongly preferred
    for audio quality — the fallback exists only as a safety net.
    """

    def __init__(self, source: AudioSource) -> None:
        """
        Args:
            source: Any AudioSource — MicrophoneStream (via adapter)
                    or FileSource.
        """
        self._source = source

    def process(self) -> np.ndarray:
        """
        Read from the source and return clean, Whisper-ready audio.

        Returns:
            np.ndarray: float32, mono, 16kHz, amplitude-normalized.
        """
        audio = self._source.read()
        sample_rate = self._source.get_sample_rate()
        channels = self._source.get_channels()

        # Emit lightweight diagnostic output for tests that capture stdout
        print(f"[PreProcessor] Input: {audio.shape} {sample_rate}Hz {channels}ch")

        audio = self._to_float32(audio)
        audio = self._to_mono(audio, channels)
        audio = self._resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        audio = self._normalize(audio)

        print(f"[PreProcessor] Output: {len(audio)} samples {TARGET_SAMPLE_RATE}Hz")

        return audio

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        channels: int = 1,
    ) -> np.ndarray:
        """
        Normalize a raw numpy array directly without an AudioSource.

        Used by StreamingTranscriber to process snapshot arrays
        that come from MicrophoneStream.get_snapshot() without
        needing to wrap them in an AudioSource object.

        Args:
            audio:       Raw float32 array from any source.
            sample_rate: Native sample rate of the input.
            channels:    Number of channels in the input.

        Returns:
            np.ndarray: float32, mono, 16kHz, amplitude-normalized.
        """
        audio = self._to_float32(audio)
        audio = self._to_mono(audio, channels)
        audio = self._resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        audio = self._normalize(audio)
        return audio

    # ------------------------------------------------------------------
    # Private — each step is isolated and unit-testable independently
    # ------------------------------------------------------------------

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        """Cast to float32 if needed."""
        if audio.dtype != np.float32:
            return audio.astype(np.float32)
        return audio

    def _to_mono(self, audio: np.ndarray, channels: int) -> np.ndarray:
        """
        Convert to mono by averaging all channels.

        Averaging is correct — it preserves loudness across channels.
        Summing would cause clipping. Picking only one channel
        would discard audio content.
        """
        if channels == 1:
            return audio.flatten()

        if audio.ndim == 2:
            # Shape (N, C) — average across channel axis
            return audio.mean(axis=1).astype(np.float32)

        # Interleaved flat array shape (N*C,) — reshape then average
        return audio.reshape(-1, channels).mean(axis=1).astype(np.float32)

    def _resample(
        self,
        audio: np.ndarray,
        from_rate: int,
        to_rate: int,
    ) -> np.ndarray:
        """
        Resample from from_rate to to_rate.

        Attempts resample_poly (scipy) first for audio quality.
        Falls back to np.interp if scipy is not available.
        """
        if from_rate == to_rate:
            return audio

        common = gcd(from_rate, to_rate)
        up = to_rate // common
        down = from_rate // common

        try:
            from scipy.signal import resample_poly

            resampled = resample_poly(audio, up, down)
        except ImportError:
            # Linear interpolation fallback — acceptable, not ideal
            original_length = len(audio)
            target_length = int(original_length * to_rate / from_rate)
            original_indices = np.linspace(0, original_length - 1, original_length)
            target_indices = np.linspace(0, original_length - 1, target_length)
            resampled = np.interp(target_indices, original_indices, audio)

        return resampled.astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize peak amplitude to 1.0.

        Skipped entirely if audio is silent (all zeros)
        to avoid division by zero producing NaN or Inf.
        """
        peak = np.max(np.abs(audio))
        if peak > 0.0:
            audio = audio / peak
        return audio
