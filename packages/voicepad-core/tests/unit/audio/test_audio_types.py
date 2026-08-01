import numpy as np
import pytest
from voicepad_core.audio import AudioWindow, RawAudio, WaveformSpec


def test_audio_window_uses_absolute_sample_positions() -> None:
    window = AudioWindow(np.zeros(480, dtype=np.float32), start_sample=1_392_000)

    assert window.end_sample == 1_392_480


def test_audio_window_samples_are_read_only() -> None:
    window = AudioWindow(np.zeros(2, dtype=np.float32), start_sample=0)

    with pytest.raises(ValueError, match="read-only"):
        window.samples[0] = 1.0


def test_waveform_spec_exposes_canonical_shape() -> None:
    spec = WaveformSpec(sample_rate=16_000, channels=1)

    assert (spec.sample_rate, spec.channels) == (16_000, 1)


@pytest.mark.parametrize(("sample_rate", "channels"), [(0, 1), (16_000, 0)])
def test_waveform_spec_rejects_invalid_metadata(sample_rate: int, channels: int) -> None:
    with pytest.raises(ValueError):
        WaveformSpec(sample_rate=sample_rate, channels=channels)


def test_raw_audio_exposes_metadata_and_duration() -> None:
    audio = RawAudio(samples=np.array([0.0, 1.0, -1.0], dtype=np.float32), sample_rate=3, channels=1)

    assert audio.duration() == pytest.approx(1.0)


def test_raw_audio_samples_are_read_only_without_copying() -> None:
    samples = np.zeros(2, dtype=np.float32)
    audio = RawAudio(samples=samples, sample_rate=16_000, channels=1)

    assert audio.samples is samples
    with pytest.raises(ValueError, match="read-only"):
        audio.samples[0] = 1.0


def test_raw_audio_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        RawAudio(samples=np.zeros(1, dtype=np.float32), sample_rate=0, channels=1)


def test_raw_audio_rejects_ambiguous_multichannel_storage() -> None:
    with pytest.raises(ValueError, match="frame-by-channel"):
        RawAudio(samples=np.zeros(4, dtype=np.float32), sample_rate=16_000, channels=2)


def test_raw_audio_counts_stereo_frames() -> None:
    audio = RawAudio(samples=np.zeros((8, 2), dtype=np.float32), sample_rate=8, channels=2)

    assert (audio.frame_count, audio.duration()) == (8, 1.0)
