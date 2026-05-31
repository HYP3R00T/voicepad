"""Tests for AudioSource abstract base class."""

import numpy as np
import pytest
from voicepad_core.audio.base import AudioSource


def test_cannot_instantiate_abstract_class():
    """AudioSource is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AudioSource()


def test_subclass_must_implement_read():
    """Subclass without read() cannot be instantiated."""

    class IncompleteSource(AudioSource):
        def get_sample_rate(self) -> int:
            return 16000

        def get_channels(self) -> int:
            return 1

    with pytest.raises(TypeError, match="read"):
        IncompleteSource()


def test_subclass_must_implement_get_sample_rate():
    """Subclass without get_sample_rate() cannot be instantiated."""

    class IncompleteSource(AudioSource):
        def read(self) -> np.ndarray:
            return np.array([0.1, 0.2], dtype=np.float32)

        def get_channels(self) -> int:
            return 1

    with pytest.raises(TypeError, match="get_sample_rate"):
        IncompleteSource()


def test_subclass_must_implement_get_channels():
    """Subclass without get_channels() cannot be instantiated."""

    class IncompleteSource(AudioSource):
        def read(self) -> np.ndarray:
            return np.array([0.1, 0.2], dtype=np.float32)

        def get_sample_rate(self) -> int:
            return 16000

    with pytest.raises(TypeError, match="get_channels"):
        IncompleteSource()


def test_valid_subclass_works():
    """Subclass with all methods implemented can be instantiated."""

    class CompleteSource(AudioSource):
        def read(self) -> np.ndarray:
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

        def get_sample_rate(self) -> int:
            return 16000

        def get_channels(self) -> int:
            return 1

    # Should instantiate successfully
    source = CompleteSource()

    # Verify it's an instance of AudioSource
    assert isinstance(source, AudioSource)

    # Verify all methods work correctly
    audio_data = source.read()
    assert audio_data is not None
    assert isinstance(audio_data, np.ndarray)
    assert audio_data.dtype == np.float32
    assert len(audio_data) == 3
    np.testing.assert_array_equal(audio_data, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    assert source.get_sample_rate() == 16000
    assert source.get_channels() == 1
