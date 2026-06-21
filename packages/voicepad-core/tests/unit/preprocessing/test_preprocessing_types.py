import numpy as np
import pytest
from voicepad_core.preprocessing.constants import TARGET_SAMPLE_RATE
from voicepad_core.preprocessing.types import PreprocessedAudio


def test_preprocessed_audio_uses_whisper_defaults() -> None:
    audio = PreprocessedAudio(samples=np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32))

    assert audio.sample_rate == TARGET_SAMPLE_RATE
    assert audio.channels == 1
    assert audio.duration() == pytest.approx(1.0)


def test_preprocessed_audio_rejects_non_mono_channels() -> None:
    with pytest.raises(ValueError, match="must be mono"):
        PreprocessedAudio(samples=np.zeros(1, dtype=np.float32), channels=2)
