from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import voicepad_core
from voicepad_core import RawAudio, transcribe_file
from voicepad_core.config import Config


def test_root_exports_owned_inference_api_only() -> None:
    """The package exposes the coordinator API without legacy global model caches."""
    assert all(
        hasattr(voicepad_core, name)
        for name in (
            "activate_model",
            "deactivate_model",
            "model_is_ready",
            "prepare_model",
            "transcribe",
            "transcribe_file",
        )
    )
    assert not any(
        hasattr(voicepad_core, name)
        for name in (
            "_model_cache",
            "ensure_model_downloaded",
            "load_model",
            "model_downloaded",
        )
    )


def test_import_does_not_configure_windows_cuda() -> None:
    """Reloading the package root does not configure native CUDA DLLs."""
    with patch("voicepad_core.inference.backends.windows_cuda.configure_windows_cuda_dlls") as configure_cuda:
        importlib.reload(voicepad_core)

    configure_cuda.assert_not_called()


def make_config(tmp_path: Path) -> Config:
    return Config(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
        transcription_model="small",
        transcription_device="cpu",
        transcription_compute_type="int8",
        language="es",
        local_agreement_file=True,
    )


def test_transcribe_file_uses_config_defaults(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFF")
    config = make_config(tmp_path)
    source = Mock()
    raw_audio = RawAudio(np.zeros(4, dtype=np.float32), sample_rate=8_000, channels=1)
    result = SimpleNamespace(text="hola")

    with (
        patch("voicepad_core.FileSource", return_value=source) as mock_source,
        patch("voicepad_core.transcribe", return_value=result) as mock_transcribe,
        patch("voicepad_core.get_config", return_value=config),
        patch("voicepad_core.postprocessing.agreement.apply_local_agreement", return_value=result) as mock_agreement,
    ):
        source.read_audio.return_value = raw_audio
        returned = transcribe_file(wav_path)

    mock_source.assert_called_once_with(wav_path)
    mock_transcribe.assert_called_once_with(
        raw_audio,
        model_name="small",
        device="cpu",
        compute_type="int8",
        language="es",
        config=config,
    )
    mock_agreement.assert_called_once_with(raw_audio, result, "small", "cpu", "int8", "es")
    assert returned is result


def test_transcribe_file_skips_local_agreement_when_disabled(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFF")
    config = make_config(tmp_path)
    object.__setattr__(config, "local_agreement_file", False)
    source = Mock()
    raw_audio = RawAudio(np.zeros(4, dtype=np.float32), sample_rate=8_000, channels=1)
    result = SimpleNamespace(text="hola")

    with (
        patch("voicepad_core.FileSource", return_value=source),
        patch("voicepad_core.transcribe", return_value=result) as mock_transcribe,
        patch("voicepad_core.get_config", return_value=config),
        patch("voicepad_core.postprocessing.agreement.apply_local_agreement") as mock_agreement,
    ):
        source.read_audio.return_value = raw_audio
        returned = transcribe_file(wav_path)

    mock_transcribe.assert_called_once()
    mock_agreement.assert_not_called()
    assert returned is result


def test_transcribe_file_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="WAV file not found"):
        transcribe_file(tmp_path / "missing.wav")
