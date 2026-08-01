from __future__ import annotations

import logging
from dataclasses import dataclass
from math import gcd

import numpy as np

from ..audio.file import AudioSource
from ..audio.types import RawAudio, WaveformSpec

logger = logging.getLogger(__name__)

MONO_CHANNELS = 1
TARGET_SAMPLE_RATE = 16_000
DEFAULT_WAVEFORM_SPEC = WaveformSpec(TARGET_SAMPLE_RATE, MONO_CHANNELS)


class PreprocessingError(Exception):
    """Base preprocessing error."""


class InvalidAudioMetadataError(ValueError, PreprocessingError):
    """Raised when audio metadata is invalid."""


class InvalidAudioShapeError(ValueError, PreprocessingError):
    """Raised when samples do not match their channel metadata."""


@dataclass(frozen=True)
class PreprocessedAudio:
    """Audio prepared to satisfy one backend input contract."""

    samples: np.ndarray
    sample_rate: int
    channels: int
    transformations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels != MONO_CHANNELS:
            raise ValueError(f"preprocessed audio must be mono, got {self.channels} channels")

    def duration(self) -> float:
        """Return the audio duration in seconds."""
        return float(self.samples.shape[0]) / self.sample_rate


class AudioPreProcessor:
    """Prepare source audio for an explicit backend input contract."""

    def __init__(self, source: AudioSource) -> None:
        self._source = source

    def process(self, target: WaveformSpec = DEFAULT_WAVEFORM_SPEC) -> PreprocessedAudio:
        return self.prepare(self._source.read_audio(), target)

    @staticmethod
    def prepare(raw_audio: RawAudio, target: WaveformSpec) -> PreprocessedAudio:
        logger.debug(
            "PreProcessor: input %s %sHz %sch",
            raw_audio.samples.shape,
            raw_audio.sample_rate,
            raw_audio.channels,
        )
        samples, transformations = AudioPreProcessor._prepare_array(
            raw_audio.samples,
            sample_rate=raw_audio.sample_rate,
            channels=raw_audio.channels,
            target=target,
        )
        logger.debug(
            "PreProcessor: output %s samples %sHz %sch transformations=%s",
            len(samples),
            target.sample_rate,
            target.channels,
            transformations,
        )
        return PreprocessedAudio(
            samples=samples,
            sample_rate=target.sample_rate,
            channels=target.channels,
            transformations=transformations,
        )

    @staticmethod
    def _prepare_array(
        audio: np.ndarray,
        sample_rate: int,
        channels: int,
        target: WaveformSpec,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        AudioPreProcessor._validate_metadata(sample_rate, channels)
        if target.channels != MONO_CHANNELS:
            raise InvalidAudioMetadataError(f"only mono backend input is supported, got {target.channels} channels")

        transformations: list[str] = []
        if audio.dtype != np.float32:
            transformations.append("float32")
        audio = AudioPreProcessor._to_float32(audio)
        if channels != target.channels:
            transformations.append("mono")
        audio = AudioPreProcessor._to_mono(audio, channels)
        if sample_rate != target.sample_rate:
            transformations.append(f"resample:{sample_rate}->{target.sample_rate}")
        audio = AudioPreProcessor._resample(audio, sample_rate, target.sample_rate)
        return np.ascontiguousarray(audio), tuple(transformations)

    @staticmethod
    def _validate_metadata(sample_rate: int, channels: int) -> None:
        if sample_rate <= 0:
            raise InvalidAudioMetadataError(f"sample_rate must be positive, got {sample_rate}")
        if channels <= 0:
            raise InvalidAudioMetadataError(f"channels must be positive, got {channels}")

    @staticmethod
    def _to_float32(audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            return audio.astype(np.float32)
        return audio

    @staticmethod
    def _to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
        if channels == MONO_CHANNELS:
            return audio.reshape(-1)
        if audio.ndim == 2:
            if audio.shape[1] != channels:
                raise InvalidAudioShapeError(f"audio declares {channels} channels but shape is {audio.shape}")
            return audio.mean(axis=1).astype(np.float32)
        if audio.size % channels != 0:
            raise InvalidAudioShapeError(f"audio with length {audio.size} cannot be reshaped into {channels} channels")
        return audio.reshape(-1, channels).mean(axis=1).astype(np.float32)

    @staticmethod
    def _resample(
        audio: np.ndarray,
        from_rate: int,
        to_rate: int,
    ) -> np.ndarray:
        if from_rate == to_rate:
            return audio

        common = gcd(from_rate, to_rate)
        up = to_rate // common
        down = from_rate // common

        from scipy.signal import resample_poly

        return resample_poly(audio, up, down).astype(np.float32)


__all__ = [
    "AudioPreProcessor",
    "DEFAULT_WAVEFORM_SPEC",
    "InvalidAudioMetadataError",
    "InvalidAudioShapeError",
    "PreprocessedAudio",
    "PreprocessingError",
    "TARGET_SAMPLE_RATE",
]
