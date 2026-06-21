import numpy as np
import pytest
from voicepad_core.audio.types import AudioFormat, RawAudio


def test_raw_audio_exposes_metadata_and_duration() -> None:
    audio = RawAudio(samples=np.array([0.0, 1.0, -1.0], dtype=np.float32), sample_rate=3, channels=1)

    assert audio.duration() == pytest.approx(1.0)
    assert audio.audio_format() == AudioFormat(sample_rate=3, channels=1)


def test_raw_audio_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        RawAudio(samples=np.zeros(1, dtype=np.float32), sample_rate=0, channels=1)


def test_audio_format_rejects_invalid_channels() -> None:
    with pytest.raises(ValueError, match="channels must be positive"):
        AudioFormat(sample_rate=16_000, channels=0)
