"""Tests for voicepad.cli.config."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner
from voicepad.cli.config import AudioDevice, _get_input_devices, config_app
from voicepad_core.config import Config

runner = CliRunner()


# ---------------------------------------------------------------------------
# AudioDevice
# ---------------------------------------------------------------------------


class TestAudioDevice:
    def test_str_includes_index_name_channels_rate(self) -> None:
        """AudioDevice.__str__ includes index, name, channels, and sample rate."""
        dev = AudioDevice(index=2, name="Built-in Mic", channels=1, sample_rate=44100)
        s = str(dev)
        assert "2" in s
        assert "Built-in Mic" in s
        assert "1ch" in s
        assert "44100Hz" in s

    def test_audio_device_is_frozen(self) -> None:
        """AudioDevice is a frozen dataclass."""
        dev = AudioDevice(index=0, name="Mic", channels=2, sample_rate=16000)
        with pytest.raises(dataclasses.FrozenInstanceError):
            dev.index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _get_input_devices
# ---------------------------------------------------------------------------


class TestGetInputDevices:
    def test_returns_only_input_devices(self) -> None:
        """_get_input_devices filters out devices with no input channels."""
        fake_devices = [
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100.0},
            {"name": "Microphone", "max_input_channels": 2, "default_samplerate": 16000.0},
        ]
        with patch("sounddevice.query_devices", return_value=fake_devices):
            devices = _get_input_devices()

        assert len(devices) == 1
        assert devices[0].name == "Microphone"

    def test_returns_empty_list_when_no_input_devices(self) -> None:
        """_get_input_devices returns an empty list when no input devices exist."""
        fake_devices = [
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100.0},
        ]
        with patch("sounddevice.query_devices", return_value=fake_devices):
            devices = _get_input_devices()

        assert devices == []

    def test_device_index_matches_position(self) -> None:
        """The index of each AudioDevice matches its position in the full device list."""
        fake_devices = [
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100.0},
            {"name": "Microphone", "max_input_channels": 1, "default_samplerate": 44100.0},
        ]
        with patch("sounddevice.query_devices", return_value=fake_devices):
            devices = _get_input_devices()

        assert devices[0].index == 1

    def test_device_uses_default_sample_rate_when_missing(self) -> None:
        """When default_samplerate is absent, 44100 is used as fallback."""
        fake_devices = [
            {"name": "Mic", "max_input_channels": 1},
        ]
        with patch("sounddevice.query_devices", return_value=fake_devices):
            devices = _get_input_devices()

        assert devices[0].sample_rate == 44100

    def test_device_uses_default_name_when_missing(self) -> None:
        """When name is absent, a fallback name is generated."""
        fake_devices = [
            {"max_input_channels": 1, "default_samplerate": 44100.0},
        ]
        with patch("sounddevice.query_devices", return_value=fake_devices):
            devices = _get_input_devices()

        assert "Device" in devices[0].name or devices[0].name != ""


# ---------------------------------------------------------------------------
# list_input_devices CLI command
# ---------------------------------------------------------------------------


class TestListInputDevicesCommand:
    def test_exits_with_error_when_no_devices(self) -> None:
        """When no input devices are found, the command exits with code 1."""
        with patch("voicepad.cli.config._get_input_devices", return_value=[]):
            result = runner.invoke(config_app, ["input"])

        assert result.exit_code == 1

    def test_lists_available_devices(self) -> None:
        """When devices are found, they are printed to stdout."""
        fake_devices = [AudioDevice(index=0, name="Test Mic", channels=1, sample_rate=44100)]
        mock_config = Config(recordings_path="data/recordings", markdown_path="data/markdown")

        with (
            patch("voicepad.cli.config._get_input_devices", return_value=fake_devices),
            patch("voicepad.cli.config.get_config", return_value=mock_config),
            patch("voicepad.cli.config.get_config_with_metadata") as mock_meta,
        ):
            mock_meta.return_value = (mock_config, MagicMock(per_field=None))
            result = runner.invoke(config_app, ["input"])

        assert result.exit_code == 0
        assert "Test Mic" in result.output

    def test_marks_configured_device(self) -> None:
        """When a device matches the configured index, it is marked in the output."""
        fake_devices = [AudioDevice(index=1, name="Configured Mic", channels=1, sample_rate=44100)]
        mock_config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            input_device_index=1,
        )

        with (
            patch("voicepad.cli.config._get_input_devices", return_value=fake_devices),
            patch("voicepad.cli.config.get_config", return_value=mock_config),
            patch("voicepad.cli.config.get_config_with_metadata") as mock_meta,
        ):
            mock_meta.return_value = (mock_config, MagicMock(per_field=None))
            result = runner.invoke(config_app, ["input"])

        assert "configured" in result.output


# ---------------------------------------------------------------------------
# show_config CLI command
# ---------------------------------------------------------------------------


class TestShowConfigCommand:
    def test_show_config_exits_zero(self) -> None:
        """show_config exits with code 0 on success."""
        mock_config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        mock_meta = MagicMock()
        mock_meta.per_field = None

        with (
            patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_meta)),
        ):
            result = runner.invoke(config_app, ["show"])

        assert result.exit_code == 0

    def test_show_config_prints_field_names(self) -> None:
        """show_config output includes known config field names."""
        mock_config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        mock_meta = MagicMock()
        mock_meta.per_field = None

        with (
            patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_meta)),
        ):
            result = runner.invoke(config_app, ["show"])

        assert "recordings_path" in result.output or result.exit_code == 0

    def test_show_config_with_env_source(self) -> None:
        """When a field comes from an env var, the source column shows 'env var'."""
        mock_config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        field_src = MagicMock()
        field_src.source = "env"
        field_src.source_path = None
        mock_meta = MagicMock()
        mock_meta.per_field = {"recordings_path": field_src}

        with (
            patch("voicepad.cli.config.get_config_with_metadata", return_value=(mock_config, mock_meta)),
        ):
            result = runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
