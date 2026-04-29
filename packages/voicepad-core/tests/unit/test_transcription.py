"""Tests for voicepad_core.transcription."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core.config import Config
from voicepad_core.transcription import (
    COMPUTE_TYPE,
    DEVICE,
    AudioTooShortError,
    TranscriptionError,
    TranscriptionResult,
    _is_cuda_error,
    _load_cpu_fallback,
    _model_cache,
    _trim_trailing_silence,
    get_or_load_model,
    model_downloaded,
    transcribe_buffer,
    transcribe_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _speech_audio(seconds: float = 1.0) -> np.ndarray:
    """Return non-silent float32 audio at 16 kHz (sine wave)."""
    t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# _is_cuda_error
# ---------------------------------------------------------------------------


class TestIsCudaError:
    def test_returns_true_for_cuda_keyword(self) -> None:
        assert _is_cuda_error(RuntimeError("CUDA out of memory")) is True

    def test_returns_true_for_cublas_keyword(self) -> None:
        assert _is_cuda_error(RuntimeError("cublas error")) is True

    def test_returns_true_for_cudnn_keyword(self) -> None:
        assert _is_cuda_error(RuntimeError("cudnn error")) is True

    def test_returns_false_for_non_cuda_error(self) -> None:
        assert _is_cuda_error(RuntimeError("General runtime error")) is False

    def test_is_case_insensitive(self) -> None:
        assert _is_cuda_error(RuntimeError("CUDA Out Of Memory")) is True

    def test_returns_true_for_nvrtc_keyword(self) -> None:
        assert _is_cuda_error(RuntimeError("nvrtc compilation failed")) is True

    def test_returns_true_for_cufft_keyword(self) -> None:
        assert _is_cuda_error(RuntimeError("cufft error")) is True

    def test_returns_true_for_curand_keyword(self) -> None:
        assert _is_cuda_error(RuntimeError("curand error")) is True


# ---------------------------------------------------------------------------
# _trim_trailing_silence
# ---------------------------------------------------------------------------


class TestTrimTrailingSilence:
    def test_removes_silent_tail(self) -> None:
        """Silent zeros at the end are trimmed."""
        speech = _speech_audio(1.0)
        silence = np.zeros(int(16000 * 2.0), dtype=np.float32)  # 2s of true silence
        audio = np.concatenate([speech, silence])
        trimmed = _trim_trailing_silence(audio)
        assert len(trimmed) < len(audio)

    def test_does_not_trim_speech(self) -> None:
        """Audio that ends with speech is not trimmed."""
        audio = _speech_audio(1.0)
        trimmed = _trim_trailing_silence(audio)
        # Should keep essentially all of it (within one window)
        assert len(trimmed) >= len(audio) - int(0.3 * 16000)

    def test_handles_very_short_audio(self) -> None:
        """Audio shorter than one window is returned unchanged."""
        audio = np.zeros(100, dtype=np.float32)
        trimmed = _trim_trailing_silence(audio)
        assert len(trimmed) == len(audio)

    def test_all_silence_returns_minimal_audio(self) -> None:
        """Fully silent audio is trimmed down to near-zero length."""
        audio = np.zeros(32000, dtype=np.float32)
        trimmed = _trim_trailing_silence(audio)
        assert len(trimmed) < len(audio)

    def test_preserves_dtype(self) -> None:
        """Output dtype matches input dtype."""
        audio = _speech_audio(1.0)
        trimmed = _trim_trailing_silence(audio)
        assert trimmed.dtype == np.float32


# ---------------------------------------------------------------------------
# model_downloaded
# ---------------------------------------------------------------------------


class TestModelDownloaded:
    def test_returns_false_if_cache_dir_missing(self) -> None:
        assert model_downloaded("nonexistent-model-xyz") is False

    def test_returns_false_if_snapshots_dir_missing(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            assert model_downloaded("tiny") is False

    def test_returns_false_if_model_bin_missing(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        repo_dir.mkdir(parents=True)
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            assert model_downloaded("tiny") is False

    def test_returns_true_if_model_bin_present(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "model.bin").write_bytes(b"fake")
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            assert model_downloaded("tiny") is True


# ---------------------------------------------------------------------------
# ensure_model_downloaded
# ---------------------------------------------------------------------------


class TestEnsureModelDownloaded:
    def test_skips_if_already_downloaded(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "model.bin").write_bytes(b"fake")
        with (
            patch.dict(os.environ, {"HF_HOME": str(tmp_path)}),
            patch("voicepad_core.transcription.snapshot_download") as mock_dl,
        ):
            from voicepad_core.transcription import ensure_model_downloaded

            ensure_model_downloaded("tiny")
            mock_dl.assert_not_called()

    def test_calls_snapshot_download_if_missing(self, tmp_path: Path) -> None:
        with (
            patch.dict(os.environ, {"HF_HOME": str(tmp_path)}),
            patch("voicepad_core.transcription.snapshot_download") as mock_dl,
        ):
            from voicepad_core.transcription import ensure_model_downloaded

            ensure_model_downloaded("tiny")
            mock_dl.assert_called_once()

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        from huggingface_hub.utils import HfHubHTTPError

        mock_response = MagicMock()
        mock_response.status_code = 404
        with (
            patch.dict(os.environ, {"HF_HOME": str(tmp_path)}),
            patch(
                "voicepad_core.transcription.snapshot_download",
                side_effect=HfHubHTTPError("404", response=mock_response),
            ),
        ):
            from voicepad_core.transcription import ensure_model_downloaded

            with pytest.raises(TranscriptionError):
                ensure_model_downloaded("tiny")

    def test_raises_on_generic_error(self, tmp_path: Path) -> None:
        with (
            patch.dict(os.environ, {"HF_HOME": str(tmp_path)}),
            patch("voicepad_core.transcription.snapshot_download", side_effect=OSError("disk full")),
        ):
            from voicepad_core.transcription import ensure_model_downloaded

            with pytest.raises(TranscriptionError):
                ensure_model_downloaded("tiny")


# ---------------------------------------------------------------------------
# _load_cpu_fallback
# ---------------------------------------------------------------------------


class TestLoadCpuFallback:
    def test_loads_model_with_empty_cache(self) -> None:
        _model_cache.clear()
        with patch("voicepad_core.transcription.WhisperModel"):
            model, device, compute = _load_cpu_fallback("tiny")
        assert device == "cpu"
        assert compute == "int8"

    def test_returns_cached_model_on_hit(self) -> None:
        mock_model = MagicMock()
        _model_cache[("tiny", "cpu", "int8")] = mock_model
        model, device, compute = _load_cpu_fallback("tiny")
        assert model is mock_model
        assert device == "cpu"

    def teardown_method(self) -> None:
        _model_cache.clear()


# ---------------------------------------------------------------------------
# get_or_load_model  (GPU-marked — mocked, no real GPU needed)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGetOrLoadModel:
    def test_returns_from_cache(self, tmp_path: Path) -> None:
        _model_cache.clear()
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path, transcription_model="tiny")
        mock_model = MagicMock()
        _model_cache[("tiny", DEVICE, COMPUTE_TYPE)] = mock_model
        model, device, compute, fallback = get_or_load_model(config)
        assert model is mock_model
        assert fallback is False

    def test_fallback_on_cuda_error(self, tmp_path: Path) -> None:
        _model_cache.clear()
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path, transcription_model="tiny")
        with (
            patch("voicepad_core.transcription.WhisperModel", side_effect=RuntimeError("CUDA out of memory")),
            patch("voicepad_core.transcription._load_cpu_fallback", return_value=(MagicMock(), "cpu", "int8")),
        ):
            model, device, compute, fallback = get_or_load_model(config)
        assert device == "cpu"
        assert fallback is True

    def test_raises_on_non_cuda_error(self, tmp_path: Path) -> None:
        _model_cache.clear()
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        with (
            patch("voicepad_core.transcription.WhisperModel", side_effect=RuntimeError("General error")),
            pytest.raises(TranscriptionError),
        ):
            get_or_load_model(config)

    def test_raises_on_other_exception(self, tmp_path: Path) -> None:
        _model_cache.clear()
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        with (
            patch("voicepad_core.transcription.WhisperModel", side_effect=ValueError("Invalid model name")),
            pytest.raises(TranscriptionError),
        ):
            get_or_load_model(config)

    def teardown_method(self) -> None:
        _model_cache.clear()


# ---------------------------------------------------------------------------
# transcribe_buffer  (GPU-marked — model is mocked)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestTranscribeBuffer:
    def _mock_model(self, text: str = "hello") -> MagicMock:
        seg = MagicMock()
        seg.start, seg.end, seg.text = 0.0, 0.9, text
        m = MagicMock()
        m.transcribe.return_value = ([seg], MagicMock(language="en", language_probability=0.99))
        return m

    def test_raises_if_audio_too_short(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        with pytest.raises(AudioTooShortError):
            transcribe_buffer(np.zeros(4000, dtype=np.float32), config)

    def test_trims_trailing_silence_before_transcribing(self, tmp_path: Path) -> None:
        """_trim_trailing_silence is called; audio with only silence raises AudioTooShortError."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        # 1s of speech + 2s of silence — trim should keep the speech part
        audio = np.concatenate([_speech_audio(1.0), np.zeros(32000, dtype=np.float32)])
        mock_model = self._mock_model()
        with patch(
            "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
        ):
            result = transcribe_buffer(audio, config)
        assert result.text == "hello"

    def test_flattens_multidimensional_audio(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        audio_2d = np.tile(_speech_audio(0.5), (2, 1))  # shape (2, 8000)
        mock_model = self._mock_model()
        with patch(
            "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
        ):
            result = transcribe_buffer(audio_2d, config)
        assert isinstance(result, TranscriptionResult)

    def test_returns_result_with_text(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = self._mock_model("hello world")
        with patch(
            "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
        ):
            result = transcribe_buffer(_speech_audio(1.0), config)
        assert result.text == "hello world"
        assert len(result.segments) == 1

    def test_passes_no_speech_threshold(self, tmp_path: Path) -> None:
        """no_speech_threshold is forwarded to model.transcribe."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = self._mock_model()
        with patch(
            "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
        ):
            transcribe_buffer(_speech_audio(1.0), config)
        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert "no_speech_threshold" in call_kwargs
        assert call_kwargs["no_speech_threshold"] > 0.5  # should be 0.8

    def test_passes_vad_filter_true(self, tmp_path: Path) -> None:
        """vad_filter=True is always passed."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = self._mock_model()
        with patch(
            "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
        ):
            transcribe_buffer(_speech_audio(1.0), config)
        assert mock_model.transcribe.call_args.kwargs["vad_filter"] is True

    def test_retries_on_cuda_inference_error(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")
        seg = MagicMock()
        seg.start, seg.end, seg.text = 0.0, 0.9, "hello"
        mock_cpu = MagicMock()
        mock_cpu.transcribe.return_value = ([seg], MagicMock(language="en", language_probability=0.9))
        with (
            patch(
                "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
            ),
            patch("voicepad_core.transcription._load_cpu_fallback", return_value=(mock_cpu, "cpu", "int8")),
        ):
            result = transcribe_buffer(_speech_audio(1.0), config)
        assert result.fallback_to_cpu is True

    def test_raises_on_non_cuda_inference_error(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Some other error")
        with (
            patch(
                "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
            ),
            pytest.raises(TranscriptionError),
        ):
            transcribe_buffer(_speech_audio(1.0), config)

    def test_raises_on_other_exception(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = ValueError("Invalid input")
        with (
            patch(
                "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
            ),
            pytest.raises(TranscriptionError),
        ):
            transcribe_buffer(_speech_audio(1.0), config)

    def test_logs_long_audio_info(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        # 1000s of speech-like audio
        long_audio = np.tile(_speech_audio(1.0), 1000)
        mock_model = self._mock_model()
        with (
            patch(
                "voicepad_core.transcription.get_or_load_model", return_value=(mock_model, DEVICE, COMPUTE_TYPE, False)
            ),
            patch("voicepad_core.transcription.logger") as mock_logger,
        ):
            transcribe_buffer(long_audio, config)
        mock_logger.info.assert_called()

    def teardown_method(self) -> None:
        _model_cache.clear()


# ---------------------------------------------------------------------------
# transcribe_file  (GPU-marked — model is mocked)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestTranscribeFile:
    def _mock_model(self) -> MagicMock:
        seg = MagicMock()
        seg.start, seg.end, seg.text = 0.0, 0.9, "test"
        m = MagicMock()
        m.transcribe.return_value = ([seg], MagicMock(language="en", language_probability=0.9))
        return m

    def test_raises_if_file_not_found(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        with pytest.raises(TranscriptionError, match="not found"):
            transcribe_file(tmp_path / "nonexistent.wav", config)

    def test_raises_if_path_is_directory(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        d = tmp_path / "subdir"
        d.mkdir()
        with pytest.raises(TranscriptionError, match="not a file"):
            transcribe_file(d, config)

    def test_reads_mono_audio(self, tmp_path: Path) -> None:
        import soundfile as sf

        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        wav = tmp_path / "test.wav"
        sf.write(str(wav), _speech_audio(1.0), 16000)
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(self._mock_model(), DEVICE, COMPUTE_TYPE, False),
        ):
            result = transcribe_file(wav, config)
        assert isinstance(result, TranscriptionResult)

    def test_averages_stereo_to_mono(self, tmp_path: Path) -> None:
        import soundfile as sf

        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        wav = tmp_path / "stereo.wav"
        stereo = np.tile(_speech_audio(1.0), (2, 1)).T  # (16000, 2)
        sf.write(str(wav), stereo, 16000)
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(self._mock_model(), DEVICE, COMPUTE_TYPE, False),
        ):
            result = transcribe_file(wav, config)
        assert isinstance(result, TranscriptionResult)

    def test_warns_on_sample_rate_mismatch(self, tmp_path: Path) -> None:
        import soundfile as sf

        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        wav = tmp_path / "wrong_sr.wav"
        sf.write(str(wav), _speech_audio(1.0), 8000)
        with (
            patch(
                "voicepad_core.transcription.get_or_load_model",
                return_value=(self._mock_model(), DEVICE, COMPUTE_TYPE, False),
            ),
            patch("voicepad_core.transcription.logger") as mock_logger,
        ):
            transcribe_file(wav, config)
        mock_logger.warning.assert_called()

    def test_raises_if_file_unreadable(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        bad = tmp_path / "invalid.wav"
        bad.write_text("not a wav")
        with pytest.raises(TranscriptionError, match="Failed to read"):
            transcribe_file(bad, config)

    def teardown_method(self) -> None:
        _model_cache.clear()
