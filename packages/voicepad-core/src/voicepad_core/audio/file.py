# audio/file.py

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import AudioSource

SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".mp4"}


class FileSource(AudioSource):
    """
    Reads audio from a file on disk.

    Supports WAV and FLAC natively via soundfile.
    Supports MP3, M4A, and other formats by converting
    to WAV first using ffmpeg (must be installed).

    The PreProcessor will resample and convert to mono afterwards.
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Args:
            file_path: Path to the audio file (WAV, MP3, FLAC, M4A etc.)
        """
        self._file_path = Path(file_path)
        self._validate()
        self._audio: np.ndarray | None = None
        self._sample_rate: int = 0
        self._channels: int = 0

    def _validate(self) -> None:
        """Check file exists and is a supported format."""
        if not self._file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self._file_path}")

        suffix = self._file_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}")

    def _load(self) -> None:
        """
        Load the file into memory.

        WAV/FLAC → soundfile (fast, no external dependency)
        MP3/M4A  → ffmpeg converts to temp WAV → soundfile reads it
        """
        suffix = self._file_path.suffix.lower()

        if suffix in {".wav", ".flac", ".ogg"}:
            # soundfile can handle these natively
            audio, sample_rate = sf.read(str(self._file_path), dtype="float32")

        else:
            # MP3, M4A etc. — convert via ffmpeg to a temp WAV
            audio, sample_rate = self._load_via_ffmpeg()

        self._sample_rate = sample_rate

        # Detect channels from shape
        if audio.ndim == 1:
            self._channels = 1
        else:
            self._channels = audio.shape[1]

        self._audio = audio

    def _load_via_ffmpeg(self) -> tuple[np.ndarray, int]:
        """Convert non-WAV file to WAV using ffmpeg, then read it."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Overwrite without asking
                    "-i",
                    str(self._file_path),
                    "-ar",
                    "44100",  # Keep original rate — PreProcessor resamples
                    "-ac",
                    "2",  # Keep original channels — PreProcessor converts
                    tmp_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio, sample_rate = sf.read(tmp_path, dtype="float32")
            return audio, sample_rate

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed to convert '{self._file_path}'. Is ffmpeg installed? Error: {e}") from e
        finally:
            # Always clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def read(self) -> np.ndarray:
        """
        Load and return all audio from the file.

        Returns:
            np.ndarray: float32 array of shape (N,) for mono
                        or (N, C) for multi-channel audio.
        """
        if self._audio is None:
            print(f"[FileSource] Loading: {self._file_path.name}")
            self._load()
            if self._audio is None:
                raise RuntimeError("_load() failed to populate audio data")
            print(f"[FileSource] Loaded {len(self._audio)} samples at {self._sample_rate}Hz, {self._channels}ch.")

        return self._audio

    def get_sample_rate(self) -> int:
        if self._audio is None:
            self._load()
        return self._sample_rate

    def get_channels(self) -> int:
        if self._audio is None:
            self._load()
        return self._channels
