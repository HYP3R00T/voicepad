"""
Comprehensive tests for AudioPreProcessor class.

Tests cover:
- Float32 conversion (5 tests)
- Mono conversion (6 tests)
- Resampling (6 tests)
- Normalization (4 tests)
- Integration (2 tests)

Target: 80%+ coverage for preprocessor.py
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
from voicepad_core.audio.base import AudioSource
from voicepad_core.preprocessing.errors import InvalidAudioMetadataError, InvalidAudioShapeError
from voicepad_core.preprocessing.preprocessor import TARGET_SAMPLE_RATE, AudioPreProcessor

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_audio_source() -> Mock:
    """Create a mock AudioSource for testing."""
    source = Mock(spec=AudioSource)
    # Default values - can be overridden in tests
    source.read.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    source.get_sample_rate.return_value = 16000
    source.get_channels.return_value = 1
    return source


@pytest.fixture
def sample_audio_mono() -> np.ndarray:
    """1D float32 array, 1 second at 16kHz."""
    return np.random.randn(16000).astype(np.float32) * 0.5


@pytest.fixture
def sample_audio_stereo() -> np.ndarray:
    """2D float32 array (N, 2), 1 second at 44.1kHz."""
    return np.random.randn(44100, 2).astype(np.float32) * 0.5


# ============================================================================
# Float32 Conversion Tests (5 tests)
# ============================================================================


def test_to_float32_converts_int16(mock_audio_source: Mock) -> None:
    """Test int16 → float32 conversion."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.array([100, -200, 300], dtype=np.int16)

    result = preprocessor._to_float32(input_array)

    assert result.dtype == np.float32
    npt.assert_array_equal(result, np.array([100.0, -200.0, 300.0], dtype=np.float32))


def test_to_float32_converts_int32(mock_audio_source: Mock) -> None:
    """Test int32 → float32 conversion."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.array([1000, -2000, 3000], dtype=np.int32)

    result = preprocessor._to_float32(input_array)

    assert result.dtype == np.float32
    npt.assert_array_equal(result, np.array([1000.0, -2000.0, 3000.0], dtype=np.float32))


def test_to_float32_converts_float64(mock_audio_source: Mock) -> None:
    """Test float64 → float32 conversion."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    result = preprocessor._to_float32(input_array)

    assert result.dtype == np.float32
    npt.assert_array_almost_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32), decimal=6)


def test_to_float32_already_float32(mock_audio_source: Mock) -> None:
    """Test already float32 (no-op)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = preprocessor._to_float32(input_array)

    assert result.dtype == np.float32
    # Should be the same array (no conversion needed)
    npt.assert_array_equal(result, input_array)


def test_to_float32_preserves_values(mock_audio_source: Mock) -> None:
    """Test that conversion preserves array values correctly."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # Test integer types with integer values
    int_values = [0.0, 1.0, -1.0, 100.0, -100.0, 1000.0, -1000.0]
    for dtype in [np.int16, np.int32]:
        input_array = np.array(int_values, dtype=dtype)
        result = preprocessor._to_float32(input_array)

        assert result.dtype == np.float32
        npt.assert_array_almost_equal(result, int_values, decimal=5)

    # Test float64 with fractional values
    float_values = [0.0, 1.0, -1.0, 0.5, -0.5, 100.123, -100.456]
    input_array = np.array(float_values, dtype=np.float64)
    result = preprocessor._to_float32(input_array)

    assert result.dtype == np.float32
    npt.assert_array_almost_equal(result, float_values, decimal=5)


# ============================================================================
# Mono Conversion Tests (6 tests)
# ============================================================================


def test_to_mono_already_mono_1d(mock_audio_source: Mock) -> None:
    """Test already mono (1D array)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = preprocessor._to_mono(input_array, channels=1)

    assert result.ndim == 1
    assert result.dtype == np.float32
    npt.assert_array_equal(result, input_array)


def test_to_mono_already_mono_2d(mock_audio_source: Mock) -> None:
    """Test already mono (2D array with 1 channel)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

    result = preprocessor._to_mono(input_array, channels=1)

    assert result.ndim == 1
    assert result.dtype == np.float32
    npt.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_to_mono_stereo_2d_shape(mock_audio_source: Mock) -> None:
    """Test stereo (N, 2) shape → averaging."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create stereo with known values: left=[1, 2, 3], right=[3, 4, 5]
    input_array = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]], dtype=np.float32)

    result = preprocessor._to_mono(input_array, channels=2)

    assert result.ndim == 1
    assert result.dtype == np.float32
    # Average: [(1+3)/2, (2+4)/2, (3+5)/2] = [2, 3, 4]
    npt.assert_array_almost_equal(result, np.array([2.0, 3.0, 4.0], dtype=np.float32), decimal=5)


def test_to_mono_multichannel(mock_audio_source: Mock) -> None:
    """Test multi-channel (N, 4) shape → averaging."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create 4-channel audio
    input_array = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )

    result = preprocessor._to_mono(input_array, channels=4)

    assert result.ndim == 1
    assert result.dtype == np.float32
    # Average: [(1+2+3+4)/4, (5+6+7+8)/4] = [2.5, 6.5]
    npt.assert_array_almost_equal(result, np.array([2.5, 6.5], dtype=np.float32), decimal=5)


def test_to_mono_interleaved_stereo(mock_audio_source: Mock) -> None:
    """Test interleaved stereo (flat array) → reshape + average."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Interleaved: [L1, R1, L2, R2, L3, R3]
    input_array = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0], dtype=np.float32)

    result = preprocessor._to_mono(input_array, channels=2)

    assert result.ndim == 1
    assert result.dtype == np.float32
    # After reshape to (3, 2): [[1, 3], [2, 4], [3, 5]]
    # Average: [2, 3, 4]
    npt.assert_array_almost_equal(result, np.array([2.0, 3.0, 4.0], dtype=np.float32), decimal=5)


def test_to_mono_ternary_operator_paths(mock_audio_source: Mock) -> None:
    """Test ternary operator path (audio.ndim == 2 vs else)."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # Path 1: audio.ndim == 2 (2D array)
    input_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result_2d = preprocessor._to_mono(input_2d, channels=2)
    npt.assert_array_almost_equal(result_2d, np.array([1.5, 3.5], dtype=np.float32), decimal=5)

    # Path 2: else (1D interleaved array)
    input_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result_1d = preprocessor._to_mono(input_1d, channels=2)
    npt.assert_array_almost_equal(result_1d, np.array([1.5, 3.5], dtype=np.float32), decimal=5)


# ============================================================================
# Resampling Tests (6 tests)
# ============================================================================


def test_resample_no_resampling_needed(mock_audio_source: Mock) -> None:
    """Test no resampling needed (16kHz → 16kHz)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    result = preprocessor._resample(input_array, from_rate=16000, to_rate=16000)

    # Should return the same array (no resampling)
    npt.assert_array_equal(result, input_array)


def test_resample_upsampling(mock_audio_source: Mock) -> None:
    """Test upsampling (8kHz → 16kHz)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create 1 second of audio at 8kHz
    input_array = np.random.randn(8000).astype(np.float32)

    result = preprocessor._resample(input_array, from_rate=8000, to_rate=16000)

    assert result.dtype == np.float32
    # Output should be approximately 2x the length (16000 samples)
    assert abs(len(result) - 16000) < 10  # Allow small tolerance


def test_resample_downsampling(mock_audio_source: Mock) -> None:
    """Test downsampling (48kHz → 16kHz)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create 1 second of audio at 48kHz
    input_array = np.random.randn(48000).astype(np.float32)

    result = preprocessor._resample(input_array, from_rate=48000, to_rate=16000)

    assert result.dtype == np.float32
    # Output should be approximately 1/3 the length (16000 samples)
    assert abs(len(result) - 16000) < 10  # Allow small tolerance


def test_resample_common_rate(mock_audio_source: Mock) -> None:
    """Test common rate (44.1kHz → 16kHz)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create 1 second of audio at 44.1kHz
    input_array = np.random.randn(44100).astype(np.float32)

    result = preprocessor._resample(input_array, from_rate=44100, to_rate=16000)

    assert result.dtype == np.float32
    # Output should be approximately 16000 samples
    assert abs(len(result) - 16000) < 10  # Allow small tolerance


def test_resample_gcd_calculation(mock_audio_source: Mock) -> None:
    """Test GCD calculation for ratio reduction."""
    from math import gcd

    # Test that GCD is calculated correctly for common rates
    # 44100 → 16000: GCD = 100, ratio = 160/441
    common = gcd(44100, 16000)
    assert common == 100
    up = 16000 // common
    down = 44100 // common
    assert up == 160
    assert down == 441

    # 48000 → 16000: GCD = 16000, ratio = 1/3
    common = gcd(48000, 16000)
    assert common == 16000
    up = 16000 // common
    down = 48000 // common
    assert up == 1
    assert down == 3


def test_resample_output_length_correct(mock_audio_source: Mock) -> None:
    """Test output length is correct for various rates."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    test_cases = [
        (8000, 16000, 8000, 16000),  # 1 sec @ 8kHz → 1 sec @ 16kHz
        (48000, 16000, 48000, 16000),  # 1 sec @ 48kHz → 1 sec @ 16kHz
        (22050, 16000, 22050, 16000),  # 1 sec @ 22.05kHz → 1 sec @ 16kHz
    ]

    for from_rate, to_rate, input_len, expected_len in test_cases:
        input_array = np.random.randn(input_len).astype(np.float32)
        result = preprocessor._resample(input_array, from_rate=from_rate, to_rate=to_rate)

        # Allow small tolerance due to polyphase filtering
        assert abs(len(result) - expected_len) < 10, (
            f"Failed for {from_rate}→{to_rate}: got {len(result)}, expected ~{expected_len}"
        )


# ============================================================================
# Normalization Tests (4 tests)
# ============================================================================


def test_normalize_peak_normalization(mock_audio_source: Mock) -> None:
    """Test peak normalization (max value → 1.0)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create audio with peak at 0.5
    input_array = np.array([0.1, 0.5, -0.3, 0.2], dtype=np.float32)

    result = preprocessor._normalize(input_array)

    # Peak should be normalized to 1.0
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-6)
    # Relative amplitudes should be preserved
    npt.assert_array_almost_equal(result, np.array([0.2, 1.0, -0.6, 0.4], dtype=np.float32), decimal=5)


def test_normalize_silent_audio(mock_audio_source: Mock) -> None:
    """Test silent audio (all zeros, no division by zero)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    input_array = np.zeros(100, dtype=np.float32)

    result = preprocessor._normalize(input_array)

    # Should return zeros without error
    npt.assert_array_equal(result, input_array)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_normalize_negative_peak(mock_audio_source: Mock) -> None:
    """Test negative peaks normalized correctly."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create audio with negative peak at -0.8
    input_array = np.array([0.2, -0.8, 0.4, -0.3], dtype=np.float32)

    result = preprocessor._normalize(input_array)

    # Absolute peak should be 1.0
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-6)
    # The -0.8 should become -1.0
    npt.assert_array_almost_equal(result, np.array([0.25, -1.0, 0.5, -0.375], dtype=np.float32), decimal=5)


def test_normalize_already_normalized(mock_audio_source: Mock) -> None:
    """Test already normalized audio (no change)."""
    preprocessor = AudioPreProcessor(mock_audio_source)
    # Create audio already normalized
    input_array = np.array([0.5, 1.0, -0.8, 0.3], dtype=np.float32)

    result = preprocessor._normalize(input_array)

    # Should remain the same (peak already at 1.0)
    npt.assert_array_almost_equal(result, input_array, decimal=6)


# ============================================================================
# Integration Tests (2 tests)
# ============================================================================


def test_process_full_pipeline_with_mock(mock_audio_source: Mock) -> None:
    """Test full pipeline with mock AudioSource."""
    # Configure mock to return stereo 44.1kHz audio
    stereo_audio = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=np.float32,
    )
    mock_audio_source.read.return_value = stereo_audio
    mock_audio_source.get_sample_rate.return_value = 44100
    mock_audio_source.get_channels.return_value = 2

    preprocessor = AudioPreProcessor(mock_audio_source)
    result = preprocessor.process()

    # Verify output properties
    assert result.dtype == np.float32
    assert result.ndim == 1  # Mono
    assert len(result) > 0
    # Should be normalized
    assert np.max(np.abs(result)) <= 1.0


def test_process_end_to_end_stereo_to_mono(mock_audio_source: Mock, capsys) -> None:
    """Test end-to-end: stereo 44.1kHz → mono 16kHz normalized."""
    # Create 1 second of stereo audio at 44.1kHz
    duration = 1.0
    sample_rate = 44100
    num_samples = int(duration * sample_rate)

    # Create stereo audio with known characteristics
    left_channel = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples))
    right_channel = np.sin(2 * np.pi * 880 * np.linspace(0, duration, num_samples))
    stereo_audio = np.column_stack([left_channel, right_channel]).astype(np.float32) * 0.5

    mock_audio_source.read.return_value = stereo_audio
    mock_audio_source.get_sample_rate.return_value = sample_rate
    mock_audio_source.get_channels.return_value = 2

    preprocessor = AudioPreProcessor(mock_audio_source)
    result = preprocessor.process()

    # Verify all transformations applied
    assert result.dtype == np.float32
    assert result.ndim == 1  # Mono

    # Output should be approximately 1 second at 16kHz
    expected_length = int(duration * TARGET_SAMPLE_RATE)
    assert abs(len(result) - expected_length) < 100  # Allow tolerance

    # Should be normalized to [-1.0, 1.0]
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-5)
    assert np.min(result) >= -1.0
    assert np.max(result) <= 1.0


# ============================================================================
# process_array Tests (6 tests)
# ============================================================================


def test_process_array_mono_16khz(mock_audio_source: Mock) -> None:
    """Test process_array with mono 16kHz audio (no conversion needed)."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = preprocessor.process_array(audio, sample_rate=16000, channels=1)

    # Should be normalized but otherwise unchanged
    assert result.dtype == np.float32
    assert result.ndim == 1
    assert len(result) == 4
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-6)


def test_process_array_stereo_to_mono(mock_audio_source: Mock) -> None:
    """Test process_array converts stereo to mono."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # Stereo audio (N, 2)
    audio = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5]], dtype=np.float32)
    result = preprocessor.process_array(audio, sample_rate=16000, channels=2)

    # Should be mono
    assert result.ndim == 1
    assert len(result) == 3
    # Values should be averaged: [0.2, 0.3, 0.4]
    expected = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    # After normalization (peak=0.4), multiply by 1/0.4 = 2.5
    npt.assert_array_almost_equal(result, expected * 2.5, decimal=5)


def test_process_array_resampling_44khz_to_16khz(mock_audio_source: Mock) -> None:
    """Test process_array resamples 44.1kHz to 16kHz."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # 1 second at 44.1kHz
    audio = np.random.randn(44100).astype(np.float32) * 0.5
    result = preprocessor.process_array(audio, sample_rate=44100, channels=1)

    # Should be resampled to 16kHz
    assert result.dtype == np.float32
    assert abs(len(result) - 16000) < 10  # Allow small tolerance


def test_process_array_multichannel_to_mono(mock_audio_source: Mock) -> None:
    """Test process_array converts multi-channel to mono."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # 4-channel audio
    audio = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    result = preprocessor.process_array(audio, sample_rate=16000, channels=4)

    # Should be mono
    assert result.ndim == 1
    assert len(result) == 2
    # Average: [(1+2+3+4)/4, (5+6+7+8)/4] = [2.5, 6.5]
    # After normalization (peak=6.5), multiply by 1/6.5
    expected = np.array([2.5, 6.5], dtype=np.float32) / 6.5
    npt.assert_array_almost_equal(result, expected, decimal=5)


def test_process_array_int16_conversion(mock_audio_source: Mock) -> None:
    """Test process_array converts int16 to float32."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    audio = np.array([100, 200, 300, 400], dtype=np.int16)
    result = preprocessor.process_array(audio, sample_rate=16000, channels=1)

    # Should be float32
    assert result.dtype == np.float32
    # Should be normalized
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-6)


def test_process_array_silent_audio(mock_audio_source: Mock) -> None:
    """Test process_array handles silent audio without errors."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    audio = np.zeros(1000, dtype=np.float32)
    result = preprocessor.process_array(audio, sample_rate=16000, channels=1)

    # Should return zeros without NaN or Inf
    assert result.dtype == np.float32
    assert len(result) == 1000
    npt.assert_array_equal(result, audio)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_process_array_upsampling(mock_audio_source: Mock) -> None:
    """Test process_array upsamples 8kHz to 16kHz."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # 1 second at 8kHz
    audio = np.random.randn(8000).astype(np.float32) * 0.5
    result = preprocessor.process_array(audio, sample_rate=8000, channels=1)

    # Should be upsampled to 16kHz
    assert result.dtype == np.float32
    assert abs(len(result) - 16000) < 10  # Allow small tolerance


def test_process_array_downsampling(mock_audio_source: Mock) -> None:
    """Test process_array downsamples 48kHz to 16kHz."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # 1 second at 48kHz
    audio = np.random.randn(48000).astype(np.float32) * 0.5
    result = preprocessor.process_array(audio, sample_rate=48000, channels=1)

    # Should be downsampled to 16kHz
    assert result.dtype == np.float32
    assert abs(len(result) - 16000) < 10  # Allow small tolerance


def test_process_array_interleaved_stereo(mock_audio_source: Mock) -> None:
    """Test process_array handles interleaved stereo correctly."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # Interleaved stereo: [L1, R1, L2, R2, L3, R3]
    audio = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0], dtype=np.float32)
    result = preprocessor.process_array(audio, sample_rate=16000, channels=2)

    # Should be mono
    assert result.ndim == 1
    assert len(result) == 3
    # After reshape to (3, 2) and average: [2, 3, 4]
    # After normalization (peak=4), multiply by 1/4 = 0.25
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32) * 0.25
    npt.assert_array_almost_equal(result, expected, decimal=5)


# ============================================================================
# Edge Cases & Scipy Fallback Tests
# ============================================================================


def test_resample_scipy_not_available_fallback(mock_audio_source: Mock) -> None:
    """Test resampling falls back to np.interp when scipy unavailable."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # Mock the import to raise ImportError
    import sys

    original_modules = sys.modules.copy()

    # Block scipy.signal import
    sys.modules["scipy.signal"] = None  # type: ignore

    try:
        audio = np.array([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float32)
        result = preprocessor._resample(audio, from_rate=8000, to_rate=16000)

        # Should still work (using np.interp fallback)
        assert result.dtype == np.float32
        assert len(result) > len(audio)  # Upsampled
    finally:
        # Restore original modules
        sys.modules.clear()
        sys.modules.update(original_modules)


def test_normalize_very_small_values(mock_audio_source: Mock) -> None:
    """Test normalization with very small values."""
    preprocessor = AudioPreProcessor(mock_audio_source)

    # Very small but non-zero values
    audio = np.array([1e-10, 2e-10, 3e-10], dtype=np.float32)
    result = preprocessor._normalize(audio)

    # Should normalize to peak of 1.0
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-6)
    # Relative ratios should be preserved
    assert result[2] == pytest.approx(1.0, abs=1e-6)
    assert result[1] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert result[0] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_process_array_rejects_non_positive_sample_rate(mock_audio_source: Mock) -> None:
    preprocessor = AudioPreProcessor(mock_audio_source)

    with pytest.raises(InvalidAudioMetadataError, match="sample_rate must be positive"):
        preprocessor.process_array(np.array([0.1], dtype=np.float32), sample_rate=0, channels=1)


def test_process_array_rejects_non_positive_channels(mock_audio_source: Mock) -> None:
    preprocessor = AudioPreProcessor(mock_audio_source)

    with pytest.raises(InvalidAudioMetadataError, match="channels must be positive"):
        preprocessor.process_array(np.array([0.1], dtype=np.float32), sample_rate=16_000, channels=0)


def test_to_mono_rejects_mismatched_2d_channel_shape(mock_audio_source: Mock) -> None:
    preprocessor = AudioPreProcessor(mock_audio_source)

    with pytest.raises(InvalidAudioShapeError, match="declares 2 channels"):
        preprocessor._to_mono(np.array([[0.1], [0.2]], dtype=np.float32), channels=2)


def test_to_mono_rejects_invalid_interleaved_length(mock_audio_source: Mock) -> None:
    preprocessor = AudioPreProcessor(mock_audio_source)

    with pytest.raises(InvalidAudioShapeError, match="cannot be reshaped"):
        preprocessor._to_mono(np.array([0.1, 0.2, 0.3], dtype=np.float32), channels=2)
