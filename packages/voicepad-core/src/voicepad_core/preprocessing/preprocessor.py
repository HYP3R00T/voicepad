from __future__ import annotations

import logging
from math import gcd

import numpy as np

from .constants import MONO_CHANNELS, TARGET_SAMPLE_RATE
from .errors import InvalidAudioMetadataError, InvalidAudioShapeError
from ..audio.base import AudioSource

logger = logging.getLogger(__name__)


class AudioPreProcessor:
    """
    Normalize audio from any AudioSource into Whisper-ready audio.

    Output is float32, mono, 16kHz, and peak-normalized.
    """

    def __init__(self, source: AudioSource) -> None:
        self._source = source

    def process(self) -> np.ndarray:
        """Read from the source and return normalized audio."""
        audio = self._source.read()
        sample_rate = self._source.get_sample_rate()
        channels = self._source.get_channels()
        logger.debug("PreProcessor: input %s %sHz %sch", audio.shape, sample_rate, channels)
        normalized = self.process_array(audio, sample_rate=sample_rate, channels=channels)
        logger.debug("PreProcessor: output %s samples %sHz", len(normalized), TARGET_SAMPLE_RATE)
        return normalized

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        channels: int = MONO_CHANNELS,
    ) -> np.ndarray:
        """Normalize a raw audio array directly."""
        self._validate_metadata(sample_rate, channels)
        audio = self._to_float32(audio)
        audio = self._to_mono(audio, channels)
        audio = self._resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        return self._normalize(audio)

    def _validate_metadata(self, sample_rate: int, channels: int) -> None:
        if sample_rate <= 0:
            raise InvalidAudioMetadataError(f"sample_rate must be positive, got {sample_rate}")
        if channels <= 0:
            raise InvalidAudioMetadataError(f"channels must be positive, got {channels}")

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            return audio.astype(np.float32)
        return audio

    def _to_mono(self, audio: np.ndarray, channels: int) -> np.ndarray:
        if channels == MONO_CHANNELS:
            return audio.reshape(-1)
        if audio.ndim == 2:
            if audio.shape[1] != channels:
                raise InvalidAudioShapeError(f"audio declares {channels} channels but shape is {audio.shape}")
            return audio.mean(axis=1).astype(np.float32)
        if audio.size % channels != 0:
            raise InvalidAudioShapeError(f"audio with length {audio.size} cannot be reshaped into {channels} channels")
        return audio.reshape(-1, channels).mean(axis=1).astype(np.float32)

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        if from_rate == to_rate:
            return audio

        common = gcd(from_rate, to_rate)
        up = to_rate // common
        down = from_rate // common

        try:
            from scipy.signal import resample_poly

            resampled = resample_poly(audio, up, down)
        except ImportError:
            original_length = len(audio)
            target_length = int(original_length * to_rate / from_rate)
            original_indices = np.linspace(0, original_length - 1, original_length)
            target_indices = np.linspace(0, original_length - 1, target_length)
            resampled = np.interp(target_indices, original_indices, audio)

        return resampled.astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        peak = np.max(np.abs(audio))
        if peak > 0.0:
            audio = audio / peak
        return audio
