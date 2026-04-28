"""Tests for voicepad_core.transcription."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core.config import Config
from voicepad_core.streaming import StreamingTranscriber
from voicepad_core.transcription import (
    COMPUTE_TYPE,
    DEVICE,
    AudioTooShortError,
    TranscriptionError,
    TranscriptionResult,
    _is_cuda_error,
    _load_cpu_fallback,
    _model_cache,
    get_or_load_model,
    model_downloaded,
    transcribe_buffer,
    transcribe_file,
)


class TestIsCudaError:
    def test_is_cuda_error_returns_true_for_cuda_keyword(self) -> None:
        """When exception contains 'cuda', _is_cuda_error returns True."""
        e = RuntimeError("CUDA out of memory")
        assert _is_cuda_error(e) is True

    def test_is_cuda_error_returns_true_for_cublas_keyword(self) -> None:
        """When exception contains 'cublas', _is_cuda_error returns True."""
        e = RuntimeError("cublas error")
        assert _is_cuda_error(e) is True

    def test_is_cuda_error_returns_true_for_cudnn_keyword(self) -> None:
        """When exception contains 'cudnn', _is_cuda_error returns True."""
        e = RuntimeError("cudnn error")
        assert _is_cuda_error(e) is True

    def test_is_cuda_error_returns_false_for_non_cuda_error(self) -> None:
        """When exception does not contain CUDA keywords, _is_cuda_error returns False."""
        e = RuntimeError("General runtime error")
        assert _is_cuda_error(e) is False

    def test_is_cuda_error_is_case_insensitive(self) -> None:
        """_is_cuda_error checks keywords in lowercase."""
        e = RuntimeError("CUDA Out Of Memory")
        assert _is_cuda_error(e) is True

    def test_is_cuda_error_returns_true_for_nvrtc_keyword(self) -> None:
        """When exception contains 'nvrtc', _is_cuda_error returns True."""
        e = RuntimeError("nvrtc compilation failed")
        assert _is_cuda_error(e) is True

    def test_is_cuda_error_returns_true_for_cufft_keyword(self) -> None:
        """When exception contains 'cufft', _is_cuda_error returns True."""
        e = RuntimeError("cufft error")
        assert _is_cuda_error(e) is True

    def test_is_cuda_error_returns_true_for_curand_keyword(self) -> None:
        """When exception contains 'curand', _is_cuda_error returns True."""
        e = RuntimeError("curand error")
        assert _is_cuda_error(e) is True


class TestModelDownloaded:
    def test_model_downloaded_returns_false_if_cache_dir_missing(self) -> None:
        """When the cache directory doesn't exist, model_downloaded returns False."""
        result = model_downloaded("nonexistent-model-xyz")
        assert result is False

    def test_model_downloaded_returns_false_if_snapshots_dir_missing(self, tmp_path: Path) -> None:
        """When the snapshots directory doesn't exist, model_downloaded returns False."""

        env = {"HF_HOME": str(tmp_path)}
        with patch.dict(os.environ, env):
            result = model_downloaded("tiny")
            assert result is False

    def test_model_downloaded_returns_false_if_model_bin_missing(self, tmp_path: Path) -> None:
        """When snapshots exist but model.bin is absent, model_downloaded returns False."""

        repo_dir = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        repo_dir.mkdir(parents=True)
        # No model.bin written

        env = {"HF_HOME": str(tmp_path)}
        with patch.dict(os.environ, env):
            result = model_downloaded("tiny")
            assert result is False

    def test_model_downloaded_returns_true_if_model_bin_present(self, tmp_path: Path) -> None:
        """When model.bin exists in a snapshot, model_downloaded returns True."""

        repo_dir = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "model.bin").write_bytes(b"fake")

        env = {"HF_HOME": str(tmp_path)}
        with patch.dict(os.environ, env):
            result = model_downloaded("tiny")
            assert result is True


class TestEnsureModelDownloaded:
    def test_ensure_model_downloaded_skips_if_already_downloaded(self, tmp_path: Path) -> None:
        """When model is already downloaded, ensure_model_downloaded does nothing."""

        repo_dir = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc123"
        repo_dir.mkdir(parents=True)
        (repo_dir / "model.bin").write_bytes(b"fake")

        env = {"HF_HOME": str(tmp_path)}
        with patch.dict(os.environ, env), patch("voicepad_core.transcription.snapshot_download") as mock_dl:
            from voicepad_core.transcription import ensure_model_downloaded

            ensure_model_downloaded("tiny")
            mock_dl.assert_not_called()

    def test_ensure_model_downloaded_calls_snapshot_download_if_missing(self, tmp_path: Path) -> None:
        """When model is not downloaded, snapshot_download is called."""

        env = {"HF_HOME": str(tmp_path)}
        with patch.dict(os.environ, env), patch("voicepad_core.transcription.snapshot_download") as mock_dl:
            from voicepad_core.transcription import ensure_model_downloaded

            ensure_model_downloaded("tiny")
            mock_dl.assert_called_once()

    def test_ensure_model_downloaded_raises_on_http_error(self, tmp_path: Path) -> None:
        """When snapshot_download raises HfHubHTTPError, TranscriptionError is raised."""

        from huggingface_hub.utils import HfHubHTTPError

        env = {"HF_HOME": str(tmp_path)}
        with patch.dict(os.environ, env), patch("voicepad_core.transcription.snapshot_download") as mock_dl:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_dl.side_effect = HfHubHTTPError("404", response=mock_response)
            from voicepad_core.transcription import ensure_model_downloaded

            with pytest.raises(TranscriptionError):
                ensure_model_downloaded("tiny")

    def test_ensure_model_downloaded_raises_on_generic_error(self, tmp_path: Path) -> None:
        """When snapshot_download raises any other error, TranscriptionError is raised."""

        env = {"HF_HOME": str(tmp_path)}
        with patch.dict(os.environ, env), patch("voicepad_core.transcription.snapshot_download") as mock_dl:
            mock_dl.side_effect = OSError("disk full")
            from voicepad_core.transcription import ensure_model_downloaded

            with pytest.raises(TranscriptionError):
                ensure_model_downloaded("tiny")


class TestLoadCpuFallback:
    def test_load_cpu_fallback_with_empty_cache(self) -> None:
        """When _load_cpu_fallback is called with an empty cache, a model is loaded."""
        _model_cache.clear()
        with patch("voicepad_core.transcription.WhisperModel"):
            model, device, compute = _load_cpu_fallback("tiny")
            assert device == "cpu"
            assert compute == "int8"

    def test_load_cpu_fallback_uses_cache_on_hit(self) -> None:
        """When the model is already cached, it returns the cached model."""
        mock_model = MagicMock()
        cache_key = ("tiny", "cpu", "int8")
        _model_cache[cache_key] = mock_model

        model, device, compute = _load_cpu_fallback("tiny")

        assert model is mock_model
        assert device == "cpu"
        assert compute == "int8"

    def teardown_method(self) -> None:
        """Clear the model cache after each test."""
        _model_cache.clear()


@pytest.mark.gpu
class TestGetOrLoadModel:
    def test_get_or_load_model_returns_from_cache(self, tmp_path: Path) -> None:
        """When model is in cache, it is returned without reloading."""
        _model_cache.clear()
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path, transcription_model="tiny")
        mock_model = MagicMock()
        cache_key = ("tiny", DEVICE, COMPUTE_TYPE)
        _model_cache[cache_key] = mock_model

        model, device, compute, fallback = get_or_load_model(config)

        assert model is mock_model
        assert fallback is False

    def test_get_or_load_model_fallback_on_cuda_error(self, tmp_path: Path) -> None:
        """When CUDA load fails, the model falls back to CPU."""
        _model_cache.clear()
        config = Config(
            recordings_path=tmp_path,
            markdown_path=tmp_path,
            transcription_model="tiny",
        )

        with patch("voicepad_core.transcription.WhisperModel") as mock_whisper:
            mock_whisper.side_effect = RuntimeError("CUDA out of memory")
            with patch("voicepad_core.transcription._load_cpu_fallback") as mock_fallback:
                mock_cpu = MagicMock()
                mock_fallback.return_value = (mock_cpu, "cpu", "int8")

                model, device, compute, fallback = get_or_load_model(config)

                assert device == "cpu"
                assert fallback is True

    def test_get_or_load_model_raises_on_non_cuda_error(self, tmp_path: Path) -> None:
        """When a non-CUDA RuntimeError occurs, TranscriptionError is raised."""
        _model_cache.clear()
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)

        with patch("voicepad_core.transcription.WhisperModel") as mock_whisper:
            mock_whisper.side_effect = RuntimeError("General error")

            with pytest.raises(TranscriptionError):
                get_or_load_model(config)

    def test_get_or_load_model_raises_on_other_exception(self, tmp_path: Path) -> None:
        """When any other exception occurs, TranscriptionError is raised."""
        _model_cache.clear()
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)

        with patch("voicepad_core.transcription.WhisperModel") as mock_whisper:
            mock_whisper.side_effect = ValueError("Invalid model name")

            with pytest.raises(TranscriptionError):
                get_or_load_model(config)

    def teardown_method(self) -> None:
        """Clear the model cache after each test."""
        _model_cache.clear()


@pytest.mark.gpu
class TestTranscribeBuffer:
    def test_transcribe_buffer_raises_if_audio_too_short(self, tmp_path: Path) -> None:
        """When audio is shorter than MIN_AUDIO_DURATION_S, AudioTooShortError is raised."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        # Create audio shorter than 0.5 seconds at 16 kHz
        short_audio = np.zeros(4000, dtype=np.float32)  # 0.25 seconds

        with pytest.raises(AudioTooShortError):
            transcribe_buffer(short_audio, config)

    def test_transcribe_buffer_flattens_multidimensional_audio(self, tmp_path: Path) -> None:
        """When audio is multidimensional, it is flattened before processing."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        # Create 2D audio (e.g., stereo)
        audio_2d = np.ones((2, 16000), dtype=np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", language_probability=0.9))

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            result = transcribe_buffer(audio_2d, config)

            assert isinstance(result, TranscriptionResult)

    def test_transcribe_buffer_logs_long_audio_warning(self, tmp_path: Path) -> None:
        """When audio exceeds MAX_AUDIO_DURATION_S, a log message is emitted."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        # Create audio longer than 900 seconds
        long_audio = np.zeros(16000 * 1000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", language_probability=0.9))

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            with patch("voicepad_core.transcription.logger") as mock_logger:
                transcribe_buffer(long_audio, config)
                mock_logger.info.assert_called()

    def test_transcribe_buffer_returns_result_with_text(self, tmp_path: Path) -> None:
        """When transcription succeeds, TranscriptionResult contains the text."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)  # 1 second

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "hello"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [mock_segment],
            MagicMock(language="en", language_probability=0.99),
        )

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            result = transcribe_buffer(audio, config)

            assert result.text == "hello"
            assert len(result.segments) == 1

    def test_transcribe_buffer_disables_previous_text_conditioning(self, tmp_path: Path) -> None:
        """When transcribing, the model is not conditioned on previous text."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [],
            MagicMock(language="en", language_probability=0.9),
        )

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            transcribe_buffer(audio, config)

            assert mock_model.transcribe.call_args.kwargs["beam_size"] == 3
            assert mock_model.transcribe.call_args.kwargs["condition_on_previous_text"] is False

    def test_transcribe_buffer_retries_on_cuda_inference_error(self, tmp_path: Path) -> None:
        """When CUDA fails during inference, the model retries on CPU."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

        mock_cpu_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "hello"
        mock_cpu_model.transcribe.return_value = (
            [mock_segment],
            MagicMock(language="en", language_probability=0.9),
        )

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            with patch("voicepad_core.transcription._load_cpu_fallback") as mock_fallback:
                mock_fallback.return_value = (mock_cpu_model, "cpu", "int8")

                result = transcribe_buffer(audio, config)

                assert result.fallback_to_cpu is True

    def test_transcribe_buffer_raises_on_non_cuda_inference_error(self, tmp_path: Path) -> None:
        """When a non-CUDA error occurs during inference, TranscriptionError is raised."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Some other error")

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            with pytest.raises(TranscriptionError):
                transcribe_buffer(audio, config)

    def test_transcribe_buffer_raises_on_other_exception(self, tmp_path: Path) -> None:
        """When any other exception occurs during transcription, TranscriptionError is raised."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = ValueError("Invalid input")

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            with pytest.raises(TranscriptionError):
                transcribe_buffer(audio, config)

    def teardown_method(self) -> None:
        """Clear the model cache after each test."""
        _model_cache.clear()


@pytest.mark.gpu
class TestTranscribeFile:
    def test_transcribe_file_raises_if_file_not_found(self, tmp_path: Path) -> None:
        """When the audio file doesn't exist, TranscriptionError is raised."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        nonexistent_path = tmp_path / "nonexistent.wav"

        with pytest.raises(TranscriptionError, match="not found"):
            transcribe_file(nonexistent_path, config)

    def test_transcribe_file_raises_if_path_is_not_file(self, tmp_path: Path) -> None:
        """When the path is a directory, TranscriptionError is raised."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        directory_path = tmp_path / "subdir"
        directory_path.mkdir()

        with pytest.raises(TranscriptionError, match="not a file"):
            transcribe_file(directory_path, config)

    def test_transcribe_file_reads_mono_audio(self, tmp_path: Path) -> None:
        """When a mono audio file is read, it is passed to transcribe_buffer."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        wav_path = tmp_path / "test.wav"

        # Create a dummy WAV file
        import soundfile as sf

        audio = np.zeros(16000, dtype=np.float32)
        sf.write(str(wav_path), audio, 16000)

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "test"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [mock_segment],
            MagicMock(language="en", language_probability=0.9),
        )

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            result = transcribe_file(wav_path, config)

            assert isinstance(result, TranscriptionResult)

    def test_transcribe_file_averages_stereo_audio(self, tmp_path: Path) -> None:
        """When stereo audio is read, channels are averaged to mono."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        wav_path = tmp_path / "stereo.wav"

        import soundfile as sf

        stereo_audio = np.ones((16000, 2), dtype=np.float32)
        sf.write(str(wav_path), stereo_audio, 16000)

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "stereo"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [mock_segment],
            MagicMock(language="en", language_probability=0.9),
        )

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            result = transcribe_file(wav_path, config)

            assert isinstance(result, TranscriptionResult)

    def test_transcribe_file_warns_if_sample_rate_mismatch(self, tmp_path: Path) -> None:
        """When the file sample rate is not 16000, a warning is logged."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        wav_path = tmp_path / "wrong_sr.wav"

        import soundfile as sf

        audio = np.zeros(8000, dtype=np.float32)
        sf.write(str(wav_path), audio, 8000)  # Wrong sample rate

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "wrong"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [mock_segment],
            MagicMock(language="en", language_probability=0.9),
        )

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            with patch("voicepad_core.transcription.logger") as mock_logger:
                transcribe_file(wav_path, config)

                mock_logger.warning.assert_called()

    def test_transcribe_file_raises_if_file_read_fails(self, tmp_path: Path) -> None:
        """When the audio file cannot be read, TranscriptionError is raised."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        wav_path = tmp_path / "invalid.wav"
        wav_path.write_text("not a real wav file")

        with pytest.raises(TranscriptionError, match="Failed to read"):
            transcribe_file(wav_path, config)

    def teardown_method(self) -> None:
        """Clear the model cache after each test."""
        _model_cache.clear()


class _FakeRecorder:
    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []


class TestStreamingTranscriber:
    def test_dispatch_chunk_only_seeds_first_chunk_with_prompt(self, tmp_path: Path) -> None:
        """When streaming chunks are transcribed, only the first chunk gets the initial prompt."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        transcriber = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=lambda _chunk: None,
            on_error=lambda _error: None,
        )

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [],
            MagicMock(language="en", language_probability=0.9),
        )

        with patch("voicepad_core.transcription.get_or_load_model") as mock_get_model:
            mock_get_model.return_value = (mock_model, DEVICE, COMPUTE_TYPE, False)

            transcriber._dispatch_chunk(np.zeros(16000 * 11, dtype=np.float32), is_final=False)
            transcriber._dispatch_chunk(np.zeros(16000 * 11, dtype=np.float32), is_final=True)

        first_call = mock_model.transcribe.call_args_list[0].kwargs
        second_call = mock_model.transcribe.call_args_list[1].kwargs

        assert first_call["beam_size"] == 3
        assert first_call["condition_on_previous_text"] is False
        assert first_call["initial_prompt"] is not None
        assert second_call["condition_on_previous_text"] is False
        assert second_call["initial_prompt"] is None
