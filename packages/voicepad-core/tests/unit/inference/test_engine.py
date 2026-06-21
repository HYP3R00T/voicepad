"""Tests for voicepad_core.inference.engine."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from voicepad_core.config import Config
from voicepad_core.inference.constants import MAX_AUDIO_DURATION_S, SAMPLE_RATE
from voicepad_core.inference.engine import _build_segments, _trim_trailing_silence, _vad_parameters, transcribe
from voicepad_core.inference.errors import AudioTooShortError, TranscriptionError

# ============================================================================
# _vad_parameters tests
# ============================================================================


def test_vad_parameters_allow_long_utterances() -> None:
    """Long uninterrupted utterances should not be capped at 15 seconds."""
    params = _vad_parameters()

    assert params["max_speech_duration_s"] == 29.0
    assert params["speech_pad_ms"] == 500


def test_vad_parameters_returns_dict() -> None:
    """_vad_parameters returns a dictionary with expected keys."""
    params = _vad_parameters()

    assert isinstance(params, dict)
    assert "threshold" in params
    assert "min_speech_duration_ms" in params
    assert "max_speech_duration_s" in params
    assert "min_silence_duration_ms" in params
    assert "speech_pad_ms" in params


def test_vad_parameters_values() -> None:
    """_vad_parameters returns expected values."""
    params = _vad_parameters()

    assert params["threshold"] == 0.5
    assert params["min_speech_duration_ms"] == 250
    assert params["max_speech_duration_s"] == 29.0
    assert params["min_silence_duration_ms"] == 1000
    assert params["speech_pad_ms"] == 500


def test_total_audio_duration_is_unbounded() -> None:
    """The engine should not impose a finite total speech duration cap."""
    assert float("inf") == MAX_AUDIO_DURATION_S


# ============================================================================
# _trim_trailing_silence tests
# ============================================================================


def test_trim_trailing_silence_removes_silent_frames() -> None:
    """_trim_trailing_silence removes silent frames from end."""
    # Create audio with silence at end
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    audio = np.concatenate([
        np.random.randn(1000) * 0.5,  # Speech
        np.zeros(500),  # Silence
    ]).astype(np.float32)

    result = _trim_trailing_silence(audio)

    # Result should be shorter (removed some silence)
    assert len(result) < len(audio)
    # Result should be close to 1000 (allow some tolerance for the trimming algorithm)
    assert len(result) <= 1200  # More lenient threshold


def test_trim_trailing_silence_preserves_non_silent_audio() -> None:
    """_trim_trailing_silence preserves audio without trailing silence."""
    audio = np.random.randn(1000).astype(np.float32) * 0.5

    result = _trim_trailing_silence(audio)

    # Should be similar length (within one frame)
    assert abs(len(result) - len(audio)) < 320  # One frame at 16kHz


def test_trim_trailing_silence_handles_all_silent_audio() -> None:
    """_trim_trailing_silence handles completely silent audio."""
    audio = np.zeros(1000, dtype=np.float32)

    result = _trim_trailing_silence(audio)

    # Should return very short or empty array
    assert len(result) < len(audio)


def test_trim_trailing_silence_custom_threshold() -> None:
    """_trim_trailing_silence accepts custom RMS threshold."""
    audio = np.concatenate([
        np.random.randn(1000) * 0.5,
        np.random.randn(500) * 0.005,  # Very quiet
    ]).astype(np.float32)

    # With default threshold (0.01), quiet part should be trimmed
    result_default = _trim_trailing_silence(audio)

    # With higher threshold (0.1), more should be trimmed
    result_high = _trim_trailing_silence(audio, rms_threshold=0.1)

    assert len(result_high) <= len(result_default)


# ============================================================================
# _build_segments tests
# ============================================================================


def test_build_segments_filters_out_of_bounds_segments() -> None:
    """_build_segments filters segments that start after audio duration."""
    mock_segment = Mock()
    mock_segment.start = 10.0
    mock_segment.end = 11.0
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.5
    mock_segment.no_speech_prob = 0.1
    mock_segment.words = []

    segments = _build_segments([mock_segment], duration_s=5.0, word_timestamps=False)

    # Segment starting at 10s should be filtered when duration is 5s
    assert len(segments) == 0


def test_build_segments_filters_high_no_speech_prob() -> None:
    """_build_segments filters segments with high no_speech_prob."""
    mock_segment = Mock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.5
    mock_segment.no_speech_prob = 0.8  # Above NO_SPEECH_THRESHOLD (0.6)
    mock_segment.words = []

    segments = _build_segments([mock_segment], duration_s=5.0, word_timestamps=False)

    assert len(segments) == 0


def test_build_segments_clamps_end_to_duration() -> None:
    """_build_segments clamps segment end to audio duration."""
    mock_segment = Mock()
    mock_segment.start = 4.0
    mock_segment.end = 6.0  # Beyond duration
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.5
    mock_segment.no_speech_prob = 0.1
    mock_segment.words = []

    segments = _build_segments([mock_segment], duration_s=5.0, word_timestamps=False)

    assert len(segments) == 1
    assert segments[0].end == 5.0


def test_build_segments_strips_text() -> None:
    """_build_segments strips whitespace from segment text."""
    mock_segment = Mock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "  test  "
    mock_segment.avg_logprob = -0.5
    mock_segment.no_speech_prob = 0.1
    mock_segment.words = []

    segments = _build_segments([mock_segment], duration_s=5.0, word_timestamps=False)

    assert segments[0].text == "test"


def test_build_segments_includes_word_timestamps_when_requested() -> None:
    """_build_segments includes word timestamps when word_timestamps=True."""
    mock_word = Mock()
    mock_word.word = "hello"
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_word.probability = 0.95

    mock_segment = Mock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "hello"
    mock_segment.avg_logprob = -0.5
    mock_segment.no_speech_prob = 0.1
    mock_segment.words = [mock_word]

    segments = _build_segments([mock_segment], duration_s=5.0, word_timestamps=True)

    assert len(segments) == 1
    assert len(segments[0].words) == 1
    assert segments[0].words[0].word == "hello"


def test_build_segments_excludes_words_when_not_requested() -> None:
    """_build_segments excludes words when word_timestamps=False."""
    mock_word = Mock()
    mock_word.word = "hello"

    mock_segment = Mock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "hello"
    mock_segment.avg_logprob = -0.5
    mock_segment.no_speech_prob = 0.1
    mock_segment.words = [mock_word]

    segments = _build_segments([mock_segment], duration_s=5.0, word_timestamps=False)

    assert len(segments) == 1
    assert len(segments[0].words) == 0


# ============================================================================
# transcribe tests
# ============================================================================


def test_transcribe_raises_on_too_short_audio() -> None:
    """transcribe raises AudioTooShortError for audio below minimum duration."""
    # Create audio shorter than MIN_AUDIO_DURATION_S (0.5s)
    audio = np.random.randn(int(SAMPLE_RATE * 0.3)).astype(np.float32)

    with pytest.raises(AudioTooShortError, match="below minimum"):
        transcribe(audio)


def test_transcribe_flattens_multidimensional_audio() -> None:
    """transcribe flattens multi-dimensional audio to mono."""
    # Create 2D audio array
    audio = np.random.randn(SAMPLE_RATE, 2).astype(np.float32) * 0.5

    with patch("voicepad_core.inference.engine.load") as mock_load:
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="en", language_probability=0.99))
        mock_load.return_value = mock_model

        result = transcribe(audio)

        # Should succeed without error
        assert result is not None


def test_transcribe_warns_on_non_english_language() -> None:
    """transcribe logs warning for non-English languages."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5

    with (
        patch("voicepad_core.inference.engine.load") as mock_load,
        patch("voicepad_core.inference.engine.logger") as mock_logger,
    ):
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="es", language_probability=0.99))
        mock_load.return_value = mock_model

        transcribe(audio, language="es")

        # Should log warning
        mock_logger.warning.assert_called()


def test_transcribe_uses_no_prompt_for_distil_models() -> None:
    """transcribe uses no initial_prompt for distil models."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5

    with patch("voicepad_core.inference.engine.load") as mock_load:
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="en", language_probability=0.99))
        mock_load.return_value = mock_model

        transcribe(audio, model_name="distil-large-v3")

        # Check that initial_prompt was None
        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["initial_prompt"] is None


def test_transcribe_uses_prompt_for_non_distil_models() -> None:
    """transcribe uses initial_prompt for non-distil models."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5

    with patch("voicepad_core.inference.engine.load") as mock_load:
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="en", language_probability=0.99))
        mock_load.return_value = mock_model

        transcribe(audio, model_name="turbo")

        # Check that initial_prompt was provided
        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["initial_prompt"] is not None


def test_transcribe_falls_back_to_cpu_on_cuda_error() -> None:
    """transcribe falls back to CPU when CUDA fails."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5

    with (
        patch("voicepad_core.inference.engine.load") as mock_load,
        patch("voicepad_core.inference.engine._load_cpu_fallback") as mock_cpu_fallback,
    ):
        # First call (CUDA) raises CUDA error
        mock_model_cuda = Mock()
        mock_model_cuda.transcribe.side_effect = RuntimeError("cuda error")
        mock_load.return_value = mock_model_cuda

        # CPU fallback succeeds
        mock_model_cpu = Mock()
        mock_model_cpu.transcribe.return_value = ([], Mock(language="en", language_probability=0.99))
        mock_cpu_fallback.return_value = mock_model_cpu

        result = transcribe(audio, device="cuda")

        # Should have fallen back to CPU
        assert result.fallback_to_cpu is True
        assert result.device == "cpu"


def test_transcribe_raises_on_non_cuda_runtime_error() -> None:
    """transcribe raises TranscriptionError for non-CUDA RuntimeErrors."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5

    with patch("voicepad_core.inference.engine.load") as mock_load:
        mock_model = Mock()
        mock_model.transcribe.side_effect = RuntimeError("other error")
        mock_load.return_value = mock_model

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcribe(audio)


def test_transcribe_returns_transcription_result() -> None:
    """transcribe returns properly formatted TranscriptionResult."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5

    with patch("voicepad_core.inference.engine.load") as mock_load:
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hello world"
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1
        mock_segment.words = []

        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], Mock(language="en", language_probability=0.99))
        mock_load.return_value = mock_model

        result = transcribe(audio)

        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.device == "auto"
        assert result.fallback_to_cpu is False


def test_transcribe_uses_config_defaults_when_args_omitted() -> None:
    """transcribe resolves omitted parameters from Config."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5
    config = Config(
        transcription_model="base",
        transcription_device="cpu",
        transcription_compute_type="int8",
        language="es",
        beam_size=3,
        transcription_vad_filter=True,
        initial_prompt="Custom prompt",
        hallucination_silence_threshold=1.5,
        no_speech_threshold=0.4,
        hallucination_max_repetitions=2,
        text_postprocessing_enabled=True,
    )

    with (
        patch("voicepad_core.inference.engine.get_config", return_value=config),
        patch("voicepad_core.inference.engine.load") as mock_load,
        patch(
            "voicepad_core.inference.engine.remove_hallucinations", side_effect=lambda text, max_repetitions: text
        ) as mock_remove,
    ):
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hola mundo"
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1
        mock_segment.words = []

        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], Mock(language="es", language_probability=0.99))
        mock_load.return_value = mock_model

        result = transcribe(audio)

        mock_load.assert_called_once_with("base", "cpu", "int8")
        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["language"] == "es"
        assert call_kwargs["beam_size"] == 3
        assert call_kwargs["vad_filter"] is True
        assert call_kwargs["initial_prompt"] == "Custom prompt"
        assert call_kwargs["hallucination_silence_threshold"] == 1.5
        assert call_kwargs["no_speech_threshold"] == 0.4
        mock_remove.assert_called_once_with("Hola mundo", max_repetitions=2)
        assert result.device == "cpu"


def test_transcribe_bypasses_text_postprocessing_when_disabled() -> None:
    """transcribe returns raw joined segment text when text cleanup is disabled."""
    audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.5
    config = Config(text_postprocessing_enabled=False)

    with (
        patch("voicepad_core.inference.engine.load") as mock_load,
        patch("voicepad_core.inference.engine.remove_hallucinations") as mock_remove,
        patch("voicepad_core.inference.engine.normalize") as mock_normalize,
    ):
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "  Hello world  "
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1
        mock_segment.words = []

        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], Mock(language="en", language_probability=0.99))
        mock_load.return_value = mock_model

        result = transcribe(audio, config=config)

        mock_remove.assert_not_called()
        mock_normalize.assert_not_called()
        assert result.text == "Hello world"
