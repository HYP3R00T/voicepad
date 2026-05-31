# audio/file.py

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import AudioSource

# Formats soundfile handles natively (no external tool needed)
_NATIVE_FORMATS = {".wav", ".flac", ".ogg"}

# Formats that require ffmpeg conversion first
_FFMPEG_FORMATS = {".mp3", ".m4a", ".mp4"}

SUPPORTED_FORMATS = _NATIVE_FORMATS | _FFMPEG_FORMATS


class FileSource(AudioSource):
    """
    Reads audio from a file on disk.

    Supports WAV, FLAC, OGG natively via soundfile.
    Supports MP3, M4A, MP4 by converting to a temp WAV
    via ffmpeg first (ffmpeg must be installed separately).

    Audio is loaded lazily — nothing is read until read() is called.
    AudioPreProcessor handles resampling and channel conversion.
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Args:
            file_path: Path to the audio file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        self._file_path = Path(file_path)
        self._validate()

        self._audio: np.ndarray | None = None
        self._sample_rate: int = 0
        self._channels: int = 0

    # ------------------------------------------------------------------
    # AudioSource interface
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """
        Load and return all audio from the file.

        Returns:
            np.ndarray: float32 array.
                        Shape (N,) for mono, (N, C) for multi-channel.
        """
        if self._audio is None:
            print(f"[FileSource] Loading: {self._file_path}")
            self._load()
            print("[FileSource] Loaded")

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

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Check the file exists and is a supported format."""
        if not self._file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self._file_path}")

        suffix = self._file_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}")

    def _load(self) -> None:
        """
        Load the audio file into memory.

        Native formats go directly through soundfile.
        Everything else is converted to a temp WAV via ffmpeg first.
        """
        suffix = self._file_path.suffix.lower()

        if suffix in _NATIVE_FORMATS:
            audio, sample_rate = sf.read(str(self._file_path), dtype="float32")
        else:
            audio, sample_rate = self._load_via_ffmpeg()

        self._sample_rate = sample_rate
        self._channels = 1 if audio.ndim == 1 else audio.shape[1]
        self._audio = audio.astype(np.float32)

    def _load_via_ffmpeg(self) -> tuple[np.ndarray, int]:
        """
        Convert a non-native format to WAV using ffmpeg,
        then read it with soundfile.

        ffmpeg is invoked with original rate and channels preserved —
        AudioPreProcessor handles all normalization after this point.

        Raises:
            RuntimeError: If ffmpeg is not installed or conversion fails.
        """
        tmp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # overwrite without asking
                    "-i",
                    str(self._file_path),
                    "-ar",
                    "44100",  # keep a known safe rate
                    "-ac",
                    "2",  # keep stereo if present
                    tmp_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            audio, sample_rate = sf.read(tmp_path, dtype="float32")
            return audio, sample_rate

        except FileNotFoundError as err:
            raise RuntimeError(
                "ffmpeg is not installed or not on PATH. Install ffmpeg to support MP3/M4A files."
            ) from err
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to convert '{self._file_path}'. Error: {e}") from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
