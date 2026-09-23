import math

import numpy as np
import pytest
from voicepad_core.audio.signal_health import SignalHealth


def test_empty_health_has_no_warning_or_nan() -> None:
    health = SignalHealth()
    assert health.with_samples(np.zeros((0, 1), dtype=np.float32), 16_000) is health
    assert health.rms == 0.0
    assert health.peak_dbfs == health.rms_dbfs == health.latest_rms_dbfs == -math.inf
    assert health.warnings == ()


def test_levels_are_sample_weighted_across_unequal_blocks() -> None:
    first = SignalHealth().with_samples(np.array([[0.5], [-0.5]], dtype=np.float32), 4)
    final = first.with_samples(np.zeros((6, 1), dtype=np.float32), 4)

    assert (first.frames, first.samples, first.duration_s) == (2, 2, 0.5)
    assert first.rms == first.peak == 0.5
    assert final.frames == final.samples == 8
    assert final.duration_s == 2.0
    assert final.rms == 0.25
    assert final.peak_dbfs == pytest.approx(-6.0206)
    assert final.rms_dbfs == pytest.approx(-12.0412)
    assert final.latest_rms_dbfs == -math.inf
    assert final.warnings == ()


def test_quiet_warning_requires_two_seconds_and_clears_with_signal() -> None:
    health = SignalHealth().with_samples(np.zeros(7, dtype=np.float32), 4)
    assert health.warnings == ()
    health = health.with_samples(np.zeros(1, dtype=np.float32), 4)
    assert "near-silent" in health.warnings[0]
    health = health.with_samples(np.array([0.25], dtype=np.float32), 4)
    assert health.warnings == ()


@pytest.mark.parametrize("level,quiet", [(0.0001, True), (0.01, False)])
def test_quiet_warning_for_low_noise_not_just_exact_zero(level: float, quiet: bool) -> None:
    health = SignalHealth().with_samples(np.full(32, level, dtype=np.float32), 16)
    assert bool(health.warnings) is quiet


def test_pcm_limit_warning_does_not_modify_or_normalize_samples() -> None:
    samples = np.array([-1.5, -1, -0.5, 0, 0.5, 32767 / 32768, 1.5], dtype=np.float32)
    original = samples.copy()
    health = SignalHealth().with_samples(samples, 16_000)

    np.testing.assert_array_equal(samples, original)
    assert health.pcm_limit_samples == 4
    assert health.peak == 1.5
    assert "possible clipping" in health.warnings[0]


def test_nonfinite_samples_are_reported_without_poisoning_statistics() -> None:
    health = SignalHealth().with_samples(np.array([np.nan, np.inf, -np.inf, 0.25]), 2)

    assert health.nonfinite_samples == 3
    assert health.rms == health.peak == 0.25
    assert len(health.warnings) == 1
    assert "non-finite" in health.warnings[0]
    invalid = SignalHealth().with_samples(np.array([np.nan, np.inf]), 1)
    assert invalid.rms == 0.0
    assert len(invalid.warnings) == 1  # Do not mislabel invalid samples as silence.


def test_stereo_samples_do_not_double_duration() -> None:
    health = SignalHealth().with_samples(np.full((8, 2), 0.25, dtype=np.float32), 4)
    assert (health.frames, health.samples, health.duration_s) == (8, 16, 2.0)
    assert health.rms == 0.25
