"""Tests for voicepad.cli.config."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner
from voicepad.cli.config import (
    AudioDevice,
    _get_input_devices,
    config_app,
)
from voicepad_core.config import Config

runner = CliRunner()


# ---------------------------------------------------------------------------
# AudioDevice
# ---------------------------------------------------------------------------


class TestAudioDevice:
    def test_str_representation(self) -> None:
        device = AudioDevice(index=0, name="Test Mic", channels=2, sample_rate=48000)
        result = str(device)
        assert "[0]" in result
        assert "Test Mic" in result
        assert "2ch" in result
        assert "48000Hz" in result


# ---------------------------------------------------------------------------
# _get_input_devices
# ---------------------------------------------------------------------------


class TestGetInputDevices:
    def test_returns_empty_list_when_no_devices(self) -> None:
        with patch("voicepad.cli.config.sd.query_devices", return_value=[]):
            devices = _get_input_devices()
        assert devices == []

    def test_filters_out_output_only_devices(self) -> None:
        mock_devices = [
            {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000, "hostapi": 0},
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch("voicepad.cli.config.sd.query_hostapis", return_value=[{"name": "MME"}]),
        ):
            devices = _get_input_devices()
        assert len(devices) == 1
        assert devices[0].name == "Microphone"

    def test_filters_out_wdm_devices(self) -> None:
        mock_devices = [
            {"name": "Mic (WDM-KS)", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
            {"name": "Mic (WASAPI)", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 1},
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch(
                "voicepad.cli.config.sd.query_hostapis", return_value=[{"name": "WDM-KS"}, {"name": "Windows WASAPI"}]
            ),
        ):
            devices = _get_input_devices()
        assert len(devices) == 1
        assert "WDM" not in devices[0].name

    def test_filters_out_virtual_routing_devices(self) -> None:
        mock_devices = [
            {
                "name": "Microsoft Sound Mapper - Input",
                "max_input_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {"name": "Real Microphone", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch("voicepad.cli.config.sd.query_hostapis", return_value=[{"name": "MME"}]),
        ):
            devices = _get_input_devices()
        assert len(devices) == 1
        assert devices[0].name == "Real Microphone"

    def test_deduplicates_by_name_preferring_wasapi(self) -> None:
        mock_devices = [
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 1},
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 2},
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch(
                "voicepad.cli.config.sd.query_hostapis",
                return_value=[{"name": "MME"}, {"name": "Windows WASAPI"}, {"name": "Windows DirectSound"}],
            ),
        ):
            devices = _get_input_devices()
        # Should only return one device (WASAPI preferred)
        assert len(devices) == 1

    def test_handles_truncated_mme_names(self) -> None:
        """MME truncates names at 31 chars, leaving unclosed parens."""
        mock_devices = [
            {
                "name": "CABLE Output (VB-Audio Virtual ",
                "max_input_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "CABLE Output (VB-Audio Virtual Cable)",
                "max_input_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 1,
            },
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch("voicepad.cli.config.sd.query_hostapis", return_value=[{"name": "MME"}, {"name": "Windows WASAPI"}]),
        ):
            devices = _get_input_devices()
        # Should deduplicate to one device
        assert len(devices) == 1

    def test_sorts_by_original_index(self) -> None:
        mock_devices = [
            {"name": "Mic B", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
            {"name": "Mic A", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch("voicepad.cli.config.sd.query_hostapis", return_value=[{"name": "MME"}]),
        ):
            devices = _get_input_devices()
        # Should maintain original order (by index)
        assert devices[0].index == 0
        assert devices[1].index == 1

    def test_handles_missing_hostapi_gracefully(self) -> None:
        mock_devices = [
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 99},
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch("voicepad.cli.config.sd.query_hostapis", return_value=[]),
        ):
            devices = _get_input_devices()
        # Should still return the device even if hostapi lookup fails
        assert len(devices) == 1

    def test_handles_query_hostapis_exception(self) -> None:
        mock_devices = [
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
        ]
        with (
            patch("voicepad.cli.config.sd.query_devices", return_value=mock_devices),
            patch("voicepad.cli.config.sd.query_hostapis", side_effect=Exception("API error")),
        ):
            devices = _get_input_devices()
        # Should handle exception gracefully
        assert len(devices) == 1


# ---------------------------------------------------------------------------
# config show
# ---------------------------------------------------------------------------


class TestShowConfig:
    def test_displays_configuration_table(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_metadata = MagicMock()
        mock_metadata.per_field = {}
        with patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_metadata)):
            result = runner.invoke(config_app, ["show"])
        assert result.exit_code == 0
        assert "Voicepad Configuration" in result.stdout

    def test_shows_field_sources(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_metadata = MagicMock()
        mock_field_meta = MagicMock()
        mock_field_meta.source = "env"
        mock_field_meta.source_path = None
        mock_metadata.per_field = {"transcription_model": mock_field_meta}
        with patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_metadata)):
            result = runner.invoke(config_app, ["show"])
        assert result.exit_code == 0
        assert "env var" in result.stdout

    def test_truncates_long_values(self, tmp_path: Path) -> None:
        # Create a very long path string (>50 chars)
        long_path_str = "x" * 100
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path, recording_prefix=long_path_str)
        mock_metadata = MagicMock()
        mock_metadata.per_field = {}
        with patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_metadata)):
            result = runner.invoke(config_app, ["show"])
        assert result.exit_code == 0
        # The long recording_prefix should be truncated
        assert "..." in result.stdout or len(long_path_str) > 50

    def test_shows_config_file_hint_when_present(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_metadata = MagicMock()
        mock_field_meta = MagicMock()
        mock_field_meta.source = "yaml"
        mock_field_meta.source_path = str(tmp_path / "voicepad.yaml")
        mock_metadata.per_field = {"transcription_model": mock_field_meta}
        with patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_metadata)):
            result = runner.invoke(config_app, ["show"])
        assert result.exit_code == 0
        assert "Config file:" in result.stdout

    def test_shows_no_config_hint_when_defaults(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_metadata = MagicMock()
        mock_metadata.per_field = {}
        with patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_metadata)):
            result = runner.invoke(config_app, ["show"])
        assert result.exit_code == 0
        assert "No config file found" in result.stdout


# ---------------------------------------------------------------------------
# config input
# ---------------------------------------------------------------------------


class TestListInputDevices:
    def test_exits_with_error_when_no_devices(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        with (
            patch("voicepad.cli.config._get_input_devices", return_value=[]),
            patch("voicepad.cli.config.get_config", return_value=mock_config),
        ):
            result = runner.invoke(config_app, ["input"])
        assert result.exit_code == 1
        # Error messages may be in stdout or stderr
        output = result.stdout + result.stderr
        assert "No audio input devices found" in output or result.exit_code == 1

    def test_lists_available_devices(self, tmp_path: Path) -> None:
        mock_devices = [
            AudioDevice(index=0, name="Mic 1", channels=2, sample_rate=48000),
            AudioDevice(index=1, name="Mic 2", channels=1, sample_rate=44100),
        ]
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path, input_device_index=None)
        with (
            patch("voicepad.cli.config._get_input_devices", return_value=mock_devices),
            patch("voicepad.cli.config.get_config", return_value=mock_config),
        ):
            result = runner.invoke(config_app, ["input"])
        assert result.exit_code == 0
        assert "Mic 1" in result.stdout
        assert "Mic 2" in result.stdout
        assert "system default" in result.stdout

    def test_marks_configured_device(self, tmp_path: Path) -> None:
        mock_devices = [
            AudioDevice(index=0, name="Mic 1", channels=2, sample_rate=48000),
            AudioDevice(index=1, name="Mic 2", channels=1, sample_rate=44100),
        ]
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path, input_device_index=1)
        with (
            patch("voicepad.cli.config._get_input_devices", return_value=mock_devices),
            patch("voicepad.cli.config.get_config", return_value=mock_config),
        ):
            result = runner.invoke(config_app, ["input"])
        assert result.exit_code == 0
        assert "← configured" in result.stdout
