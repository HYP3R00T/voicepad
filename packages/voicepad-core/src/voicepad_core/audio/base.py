# audio/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AudioSource(ABC):
    """
    Abstract base class for all audio sources.

    Every audio source must be able to report its native
    sample rate and channel count. What it returns from
    read() is raw — PreProcessor handles normalization.
    """

    @abstractmethod
    def read(self) -> np.ndarray:
        """
        Read audio from this source.

        Returns:
            np.ndarray: Raw float32 audio samples.
                        Sample rate and channels are NOT normalized here.
                        AudioPreProcessor handles that.
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
        Return the number of audio channels from this source.

        Returns:
            int: 1 for mono, 2 for stereo
        """
        ...
