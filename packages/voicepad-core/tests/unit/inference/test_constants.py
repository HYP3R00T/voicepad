"""Tests for voicepad_core.inference.constants."""

from __future__ import annotations

from voicepad_core.inference.constants import (
    BEAM_SIZE,
    COMPUTE_TYPE,
    CPU_COMPUTE_TYPE,
    CUDA_ERROR_KEYWORDS,
    DEFAULT_MODEL,
    DEVICE,
    DISTIL_MODELS,
    HALLUCINATION_SILENCE_THRESHOLD,
    HF_REPO_PREFIX,
    INITIAL_PROMPT,
    LANGUAGE,
    MAX_AUDIO_DURATION_S,
    MIN_AUDIO_DURATION_S,
    NO_SPEECH_THRESHOLD,
    SAMPLE_RATE,
)

# ============================================================================
# Device & precision constants
# ============================================================================


def test_device_is_cuda() -> None:
    """DEVICE constant is 'cuda'."""
    assert DEVICE == "cuda"
    assert isinstance(DEVICE, str)


def test_compute_type_is_int8_float16() -> None:
    """COMPUTE_TYPE is int8_float16 for GPU."""
    assert COMPUTE_TYPE == "int8_float16"
    assert isinstance(COMPUTE_TYPE, str)


def test_cpu_compute_type_is_int8() -> None:
    """CPU_COMPUTE_TYPE is int8 for CPU fallback."""
    assert CPU_COMPUTE_TYPE == "int8"
    assert isinstance(CPU_COMPUTE_TYPE, str)


# ============================================================================
# Model defaults
# ============================================================================


def test_default_model_is_turbo() -> None:
    """DEFAULT_MODEL is 'turbo'."""
    assert DEFAULT_MODEL == "turbo"
    assert isinstance(DEFAULT_MODEL, str)


def test_distil_models_is_frozenset() -> None:
    """DISTIL_MODELS is a frozenset."""
    assert isinstance(DISTIL_MODELS, frozenset)


def test_distil_models_contains_expected_models() -> None:
    """DISTIL_MODELS contains all expected distil model names."""
    expected = {
        "distil-small.en",
        "distil-medium.en",
        "distil-large-v2",
        "distil-large-v3",
        "distil-large-v3.5",
    }
    assert expected == DISTIL_MODELS


def test_distil_models_is_immutable() -> None:
    """DISTIL_MODELS cannot be modified (frozenset)."""
    # frozenset has no add method
    assert not hasattr(DISTIL_MODELS, "add")
    assert isinstance(DISTIL_MODELS, frozenset)


# ============================================================================
# Transcription quality constants
# ============================================================================


def test_beam_size_is_5() -> None:
    """BEAM_SIZE is 5."""
    assert BEAM_SIZE == 5
    assert isinstance(BEAM_SIZE, int)


def test_language_is_en() -> None:
    """LANGUAGE is 'en'."""
    assert LANGUAGE == "en"
    assert isinstance(LANGUAGE, str)


def test_no_speech_threshold_is_0_6() -> None:
    """NO_SPEECH_THRESHOLD is 0.6."""
    assert NO_SPEECH_THRESHOLD == 0.6
    assert isinstance(NO_SPEECH_THRESHOLD, float)


def test_hallucination_silence_threshold_is_2_0() -> None:
    """HALLUCINATION_SILENCE_THRESHOLD is 2.0."""
    assert HALLUCINATION_SILENCE_THRESHOLD == 2.0
    assert isinstance(HALLUCINATION_SILENCE_THRESHOLD, float)


def test_initial_prompt_is_string() -> None:
    """INITIAL_PROMPT is a non-empty string."""
    assert isinstance(INITIAL_PROMPT, str)
    assert len(INITIAL_PROMPT) > 0


def test_initial_prompt_contains_expected_content() -> None:
    """INITIAL_PROMPT contains expected guidance text."""
    assert "transcription" in INITIAL_PROMPT.lower()
    assert "punctuation" in INITIAL_PROMPT.lower()
    assert "capitalization" in INITIAL_PROMPT.lower()


# ============================================================================
# Audio duration guards
# ============================================================================


def test_sample_rate_is_16000() -> None:
    """SAMPLE_RATE is 16000 Hz."""
    assert SAMPLE_RATE == 16_000
    assert isinstance(SAMPLE_RATE, int)


def test_min_audio_duration_is_0_5() -> None:
    """MIN_AUDIO_DURATION_S is 0.5 seconds."""
    assert MIN_AUDIO_DURATION_S == 0.5
    assert isinstance(MIN_AUDIO_DURATION_S, float)


def test_max_audio_duration_is_infinite() -> None:
    """MAX_AUDIO_DURATION_S is infinite (no cap)."""
    assert float("inf") == MAX_AUDIO_DURATION_S
    assert isinstance(MAX_AUDIO_DURATION_S, float)


def test_min_duration_less_than_max_duration() -> None:
    """MIN_AUDIO_DURATION_S is less than MAX_AUDIO_DURATION_S."""
    assert MIN_AUDIO_DURATION_S < MAX_AUDIO_DURATION_S


# ============================================================================
# HuggingFace constants
# ============================================================================


def test_hf_repo_prefix() -> None:
    """HF_REPO_PREFIX is correct."""
    assert HF_REPO_PREFIX == "Systran/faster-whisper-"
    assert isinstance(HF_REPO_PREFIX, str)


def test_hf_repo_prefix_ends_with_dash() -> None:
    """HF_REPO_PREFIX ends with dash for concatenation."""
    assert HF_REPO_PREFIX.endswith("-")


# ============================================================================
# CUDA error detection
# ============================================================================


def test_cuda_error_keywords_is_tuple() -> None:
    """CUDA_ERROR_KEYWORDS is a tuple."""
    assert isinstance(CUDA_ERROR_KEYWORDS, tuple)


def test_cuda_error_keywords_contains_expected_keywords() -> None:
    """CUDA_ERROR_KEYWORDS contains expected CUDA library names."""
    expected = {"cublas", "cuda", "cudnn", "nvrtc", "cufft", "curand"}
    assert set(CUDA_ERROR_KEYWORDS) == expected


def test_cuda_error_keywords_all_lowercase() -> None:
    """All CUDA_ERROR_KEYWORDS are lowercase."""
    for keyword in CUDA_ERROR_KEYWORDS:
        assert keyword == keyword.lower()


def test_cuda_error_keywords_all_strings() -> None:
    """All CUDA_ERROR_KEYWORDS are strings."""
    for keyword in CUDA_ERROR_KEYWORDS:
        assert isinstance(keyword, str)


def test_cuda_error_keywords_is_immutable() -> None:
    """CUDA_ERROR_KEYWORDS cannot be modified (tuple)."""
    # Tuples are immutable
    assert isinstance(CUDA_ERROR_KEYWORDS, tuple)
    # Verify it's truly immutable by checking type
    assert type(CUDA_ERROR_KEYWORDS).__name__ == "tuple"


# ============================================================================
# Integration tests
# ============================================================================


def test_all_constants_are_defined() -> None:
    """All expected constants are defined and importable."""
    # This test ensures no constants are accidentally removed
    constants = [
        DEVICE,
        COMPUTE_TYPE,
        CPU_COMPUTE_TYPE,
        DEFAULT_MODEL,
        DISTIL_MODELS,
        BEAM_SIZE,
        LANGUAGE,
        NO_SPEECH_THRESHOLD,
        HALLUCINATION_SILENCE_THRESHOLD,
        INITIAL_PROMPT,
        SAMPLE_RATE,
        MIN_AUDIO_DURATION_S,
        MAX_AUDIO_DURATION_S,
        HF_REPO_PREFIX,
        CUDA_ERROR_KEYWORDS,
    ]

    # All should be defined (not None)
    for constant in constants:
        assert constant is not None


def test_numeric_constants_have_valid_values() -> None:
    """Numeric constants have sensible values."""
    assert BEAM_SIZE > 0
    assert SAMPLE_RATE > 0
    assert MIN_AUDIO_DURATION_S > 0
    assert 0.0 < NO_SPEECH_THRESHOLD < 1.0
    assert HALLUCINATION_SILENCE_THRESHOLD > 0
