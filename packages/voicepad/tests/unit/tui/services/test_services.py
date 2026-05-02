"""Tests for TUI service helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core.config import Config
from voicepad_core.transcription import AudioTooShortError, TranscriptionError


def _make_config(tmp_path: Path) -> Config:
    return Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")


class TestSettingsService:
    def test_init_stores_config(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)

        assert service.config is config

    def test_get_config_path_uses_utilityhub_config(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)

        expected_path = tmp_path / "voicepad.yaml"
        with patch("voicepad.tui.services.settings_service.get_config_path", return_value=expected_path):
            assert service.get_config_path() == expected_path

    def test_config_exists_checks_path(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)
        config_path = tmp_path / "voicepad.yaml"
        config_path.write_text("{}")

        with patch.object(service, "get_config_path", return_value=config_path):
            assert service.config_exists() is True

    def test_save_config_writes_config(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)
        config_path = tmp_path / "nested" / "voicepad.yaml"

        with (
            patch.object(service, "get_config_path", return_value=config_path),
            patch("voicepad.tui.services.settings_service.write_config") as mock_write_config,
        ):
            service.save_config(config)

        assert config_path.parent.exists()
        mock_write_config.assert_called_once_with(config, "voicepad", path=config_path, format="yaml")

    def test_save_config_raises_when_write_fails(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)
        config_path = tmp_path / "voicepad.yaml"

        with (
            patch.object(service, "get_config_path", return_value=config_path),
            patch("voicepad.tui.services.settings_service.write_config", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError),
        ):
            service.save_config(config)

    def test_update_field_returns_updated_config(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)

        with patch.object(service, "save_config") as mock_save_config:
            updated = service.update_field("recording_prefix", "notes")

        assert updated.recording_prefix == "notes"
        mock_save_config.assert_called_once_with(updated)

    def test_update_field_raises_for_invalid_value(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)

        with pytest.raises(ValueError):
            service.update_field("transcription_model", "not-a-real-model")

    def test_update_fields_returns_updated_config(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)

        with patch.object(service, "save_config") as mock_save_config:
            updated = service.update_fields({"recording_prefix": "draft", "global_hotkey": "<ctrl>+<alt>+r"})

        assert updated.recording_prefix == "draft"
        assert updated.global_hotkey == "<ctrl>+<alt>+r"
        mock_save_config.assert_called_once_with(updated)

    def test_validate_field_returns_true_for_valid_value(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)

        is_valid, error = service.validate_field("recording_prefix", "meeting")

        assert is_valid is True
        assert error is None

    def test_validate_field_returns_false_for_invalid_value(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = _make_config(tmp_path)
        service = SettingsService(config)

        is_valid, error = service.validate_field("transcription_model", "nope")

        assert is_valid is False
        assert error is not None


class TestTranscriptionService:
    def test_init_stores_config(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)

        assert service.config is config

    def test_warm_model_downloads_and_loads(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)

        with (
            patch("voicepad.tui.services.transcription_service.model_downloaded", return_value=False),
            patch("voicepad.tui.services.transcription_service.ensure_model_downloaded") as mock_download,
            patch(
                "voicepad.tui.services.transcription_service.get_or_load_model",
                return_value=(MagicMock(), "cuda", "float16", False),
            ),
        ):
            result = service.warm_model()

        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.fallback is False
        mock_download.assert_called_once()

    def test_warm_model_returns_fallback_on_transcription_error(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)

        with (
            patch("voicepad.tui.services.transcription_service.model_downloaded", return_value=True),
            patch(
                "voicepad.tui.services.transcription_service.get_or_load_model",
                side_effect=TranscriptionError("load failed"),
            ),
        ):
            result = service.warm_model()

        assert result.device == "cpu"
        assert result.fallback is True
        assert result.error == "load failed"

    def test_warm_model_returns_fallback_on_unexpected_error(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)

        with (
            patch("voicepad.tui.services.transcription_service.model_downloaded", return_value=True),
            patch("voicepad.tui.services.transcription_service.get_or_load_model", side_effect=RuntimeError("boom")),
        ):
            result = service.warm_model()

        assert result.device == "cpu"
        assert result.fallback is True
        assert result.error == "boom"

    def test_transcribe_audio_returns_result(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)
        transcription_result = MagicMock(text="hello world", latency_ms=42.0)

        with patch(
            "voicepad.tui.services.transcription_service.transcribe_buffer", return_value=transcription_result
        ) as mock_transcribe:
            result = service.transcribe_audio(np.zeros(16000, dtype=np.float32))

        assert result is transcription_result
        mock_transcribe.assert_called_once()

    @pytest.mark.parametrize(
        "exception_type",
        [AudioTooShortError, TranscriptionError, RuntimeError],
    )
    def test_transcribe_audio_raises_for_errors(self, tmp_path: Path, exception_type: type[Exception]) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)

        with (
            patch("voicepad.tui.services.transcription_service.transcribe_buffer", side_effect=exception_type("boom")),
            pytest.raises(exception_type),
        ):
            service.transcribe_audio(np.zeros(16000, dtype=np.float32))

    def test_transcribe_file_raises_for_missing_path(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)

        with pytest.raises(FileNotFoundError):
            service.transcribe_file(tmp_path / "missing.wav")

    def test_transcribe_file_uses_audio_directly_at_16khz(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)
        wav_path = tmp_path / "audio.wav"
        wav_path.write_bytes(b"fake")
        transcription_result = MagicMock(text="ok", latency_ms=1.0)

        with (
            patch(
                "voicepad.tui.services.transcription_service.sf.read",
                return_value=(np.zeros(32, dtype=np.float32), 16000),
            ),
            patch.object(service, "transcribe_audio", return_value=transcription_result) as mock_transcribe_audio,
        ):
            result = service.transcribe_file(wav_path)

        assert result is transcription_result
        mock_transcribe_audio.assert_called_once()

    def test_transcribe_file_resamples_before_transcribing(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = _make_config(tmp_path)
        service = TranscriptionService(config)
        wav_path = tmp_path / "audio.wav"
        wav_path.write_bytes(b"fake")
        transcription_result = MagicMock(text="ok", latency_ms=1.0)

        with (
            patch(
                "voicepad.tui.services.transcription_service.sf.read",
                return_value=(np.array([0.0, 0.5, -0.25], dtype=np.float32), 8000),
            ),
            patch.object(service, "transcribe_audio", return_value=transcription_result) as mock_transcribe_audio,
        ):
            result = service.transcribe_file(wav_path)

        assert result is transcription_result
        mock_transcribe_audio.assert_called_once()
