from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from voicepad_core import transcribe_file
from voicepad_core.config import Config


def make_config(tmp_path: Path) -> Config:
    return Config(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
        transcription_model="base",
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
    preprocessor = Mock()
    preprocessor.process.return_value = "processed-audio"
    result = SimpleNamespace(text="hola")

    with (
        patch("voicepad_core.FileSource", return_value=source) as mock_source,
        patch("voicepad_core.AudioPreProcessor", return_value=preprocessor) as mock_preprocessor,
        patch("voicepad_core.transcribe", return_value=result) as mock_transcribe,
        patch("voicepad_core.get_config", return_value=config),
        patch("voicepad_core.postprocessing.agreement.apply_local_agreement", return_value=result) as mock_agreement,
    ):
        source.read.return_value = "raw-audio"
        returned = transcribe_file(wav_path)

    mock_source.assert_called_once_with(wav_path)
    mock_preprocessor.assert_called_once_with(source)
    mock_transcribe.assert_called_once_with(
        "processed-audio",
        model_name="base",
        device="cpu",
        compute_type="int8",
        language="es",
        config=config,
    )
    mock_agreement.assert_called_once_with("processed-audio", result, "base", "cpu", "int8", "es")
    assert returned is result


def test_transcribe_file_skips_local_agreement_when_disabled(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFF")
    config = make_config(tmp_path)
    object.__setattr__(config, "local_agreement_file", False)
    source = Mock()
    preprocessor = Mock()
    preprocessor.process.return_value = "processed-audio"
    result = SimpleNamespace(text="hola")

    with (
        patch("voicepad_core.FileSource", return_value=source),
        patch("voicepad_core.AudioPreProcessor", return_value=preprocessor),
        patch("voicepad_core.transcribe", return_value=result) as mock_transcribe,
        patch("voicepad_core.get_config", return_value=config),
        patch("voicepad_core.postprocessing.agreement.apply_local_agreement") as mock_agreement,
    ):
        source.read.return_value = "raw-audio"
        returned = transcribe_file(wav_path)

    mock_transcribe.assert_called_once()
    mock_agreement.assert_not_called()
    assert returned is result


def test_transcribe_file_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="WAV file not found"):
        transcribe_file(tmp_path / "missing.wav")
