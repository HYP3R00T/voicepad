from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import FFMPEG_COMMAND, NATIVE_FORMATS, SUPPORTED_FORMATS
from .errors import (
    AudioConversionDependencyError,
    AudioConversionError,
    AudioFileNotFoundError,
    UnsupportedAudioFormatError,
)
from .types import RawAudio

logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """Source that provides samples and their native format."""

    @abstractmethod
    def read(self) -> np.ndarray: ...

    @abstractmethod
    def get_sample_rate(self) -> int: ...

    @abstractmethod
    def get_channels(self) -> int: ...

    def read_audio(self) -> RawAudio:
        return RawAudio(self.read(), self.get_sample_rate(), self.get_channels())


class FileSource(AudioSource):
    """Read raw audio from a file on disk."""

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._validate()
        self._audio: np.ndarray | None = None
        self._sample_rate = 0
        self._channels = 0

    def read(self) -> np.ndarray:
        if self._audio is None:
            logger.debug("FileSource loading %s", self._file_path)
            self._load()
            logger.debug("FileSource loaded %s", self._file_path)
        assert self._audio is not None
        return self._audio

    def get_sample_rate(self) -> int:
        if self._audio is None:
            self._load()
        return self._sample_rate

    def get_channels(self) -> int:
        if self._audio is None:
            self._load()
        return self._channels

    def _validate(self) -> None:
        if not self._file_path.exists():
            raise AudioFileNotFoundError(f"Audio file not found: {self._file_path}")

        suffix = self._file_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise UnsupportedAudioFormatError(f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}")

    def _load(self) -> None:
        suffix = self._file_path.suffix.lower()
        audio, sample_rate = (
            sf.read(str(self._file_path), dtype="float32") if suffix in NATIVE_FORMATS else self._load_via_ffmpeg()
        )
        self._sample_rate = sample_rate
        self._channels = 1 if audio.ndim == 1 else audio.shape[1]
        self._audio = audio.astype(np.float32)

    def _load_via_ffmpeg(self) -> tuple[np.ndarray, int]:
        tmp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            subprocess.run(
                [FFMPEG_COMMAND, "-y", "-i", str(self._file_path), tmp_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return sf.read(tmp_path, dtype="float32")
        except FileNotFoundError as err:
            raise AudioConversionDependencyError(
                f"{FFMPEG_COMMAND} is not installed or not on PATH. Install ffmpeg to support MP3/M4A files."
            ) from err
        except subprocess.CalledProcessError as err:
            raise AudioConversionError(f"ffmpeg failed to convert '{self._file_path}'. Error: {err}") from err
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
