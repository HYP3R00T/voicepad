from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
from voicepad_core.audio import AudioSource, RawAudio, WaveformSpec
from voicepad_core.preprocessing import (
    DEFAULT_WAVEFORM_SPEC,
    TARGET_SAMPLE_RATE,
    AudioPreProcessor,
    InvalidAudioMetadataError,
    InvalidAudioShapeError,
)


def prepare(samples: np.ndarray, sample_rate: int, channels: int = 1) -> np.ndarray:
    audio = RawAudio(samples, sample_rate=sample_rate, channels=channels)
    return AudioPreProcessor.prepare(audio, DEFAULT_WAVEFORM_SPEC).samples


def test_prepare_preserves_mono_float32_amplitudes() -> None:
    samples = np.array([0.1, -0.5, 0.25], dtype=np.float32)

    result = prepare(samples, TARGET_SAMPLE_RATE)

    npt.assert_array_equal(result, samples)
    assert result.flags.c_contiguous


def test_prepare_converts_stereo_to_mono_without_normalizing() -> None:
    samples = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5]], dtype=np.float32)

    result = prepare(samples, TARGET_SAMPLE_RATE, channels=2)

    npt.assert_allclose(result, np.array([0.2, 0.3, 0.4], dtype=np.float32))


def test_prepare_converts_to_float32() -> None:
    result = prepare(np.array([1.0, -2.0], dtype=np.float64), TARGET_SAMPLE_RATE)

    assert result.dtype == np.float32
    npt.assert_array_equal(result, np.array([1.0, -2.0], dtype=np.float32))


@pytest.mark.parametrize("source_rate", [8_000, 44_100, 48_000])
def test_prepare_resamples_to_16khz(source_rate: int) -> None:
    samples = np.linspace(-0.5, 0.5, source_rate, dtype=np.float32)

    result = prepare(samples, source_rate)

    assert result.dtype == np.float32
    assert len(result) == TARGET_SAMPLE_RATE


def test_process_reads_audio_source() -> None:
    source = Mock(spec=AudioSource)
    samples = np.array([0.1, 0.2], dtype=np.float32)
    source.read_audio.return_value = RawAudio(samples, sample_rate=TARGET_SAMPLE_RATE, channels=1)

    result = AudioPreProcessor(source).process()

    source.read_audio.assert_called_once_with()
    npt.assert_array_equal(result.samples, samples)
    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.channels == 1
    assert result.transformations == ()


def test_prepare_records_transformations() -> None:
    samples = np.ones((48_000, 2), dtype=np.float64)
    audio = RawAudio(samples, sample_rate=48_000, channels=2)

    result = AudioPreProcessor.prepare(audio, DEFAULT_WAVEFORM_SPEC)

    assert result.transformations == ("float32", "mono", "resample:48000->16000")


def test_prepare_rejects_non_mono_target() -> None:
    audio = RawAudio(np.zeros(10, dtype=np.float32), sample_rate=TARGET_SAMPLE_RATE, channels=1)

    with pytest.raises(InvalidAudioMetadataError, match="only mono"):
        AudioPreProcessor.prepare(audio, WaveformSpec(TARGET_SAMPLE_RATE, channels=2))


def test_prepare_rejects_invalid_source_metadata() -> None:
    with pytest.raises(InvalidAudioMetadataError, match="sample_rate"):
        AudioPreProcessor._prepare_array(
            np.zeros(10, dtype=np.float32),
            sample_rate=0,
            channels=1,
            target=DEFAULT_WAVEFORM_SPEC,
        )


def test_to_mono_rejects_mismatched_channel_shape() -> None:
    with pytest.raises(InvalidAudioShapeError, match="declares 2 channels"):
        AudioPreProcessor._to_mono(np.array([[0.1], [0.2]], dtype=np.float32), channels=2)


def test_to_mono_rejects_invalid_interleaved_length() -> None:
    with pytest.raises(InvalidAudioShapeError, match="cannot be reshaped"):
        AudioPreProcessor._to_mono(np.array([0.1, 0.2, 0.3], dtype=np.float32), channels=2)
