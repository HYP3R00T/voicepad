# audio/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AudioSource(ABC):
    """
    Abstract base class for all audio sources.

    Every audio source — microphone or file — must produce
    a normalized, 16kHz, mono, float32 numpy array.
    That is the only contract downstream cares about.
    """

    @abstractmethod
    def read(self) -> np.ndarray:
        """
        Read audio from the source.

        Returns:
            np.ndarray: Raw audio samples as float32.
                        Sample rate and channels are NOT guaranteed here.
                        PreProcessor handles that normalization.
        """
        ...

    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Return the native sample rate of this source.

        Returns:
            int: Sample rate in Hz (e.g. 44100, 48000, 16000)
        """
        ...

    @abstractmethod
    def get_channels(self) -> int:
        """
        Return the number of channels from this source.

        Returns:
            int: 1 for mono, 2 for stereo
        """
        ...
