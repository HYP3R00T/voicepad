"""Unit tests for config CLI commands."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import typer
from voicepad_core.diagnostics.models import (
    CPUInfo,
    CTranslate2Result,
    GPUDiagnosticsReport,
    NvidiaCheckResult,
    RAMInfo,
    WhisperGPUResult,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voicepad-core" / "src"))

from voicepad.cli.config import (
    AudioDevice,
    _get_input_devices,
    _show_config_hint,
    list_input_devices,
    list_models,
    recommend_model,
    show_config,
    system_info,
    transcription_config,
)


class ConfigCliTests(unittest.TestCase):
    def test_audio_device_str(self) -> None:
        dev = AudioDevice(index=1, name="Mic", channels=2, sample_rate=48000)
        self.assertIn("Mic", str(dev))

    def test_get_input_devices_filters_and_defaults(self) -> None:
        devices = [
            {"name": "Out", "max_input_channels": 0},
            {"name": "Mic", "max_input_channels": 2, "default_samplerate": 48000.0},
            {"name": "Mic2", "max_input_channels": 1, "default_samplerate": 0.0},
        ]
        with patch("voicepad.cli.config.sd.query_devices", return_value=devices):
            result = _get_input_devices()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].sample_rate, 48000)
        self.assertEqual(result[1].sample_rate, 44100)

    def test_show_config_renders_table(self) -> None:
        cfg = SimpleNamespace(model_fields={"recording_prefix": None}, recording_prefix="recording")
        source = SimpleNamespace(source="defaults", source_path=None)
        metadata = SimpleNamespace(per_field={"recording_prefix": source})
        mock_table = MagicMock()
        with (
            patch("voicepad.cli.config.get_config_with_metadata", return_value=(cfg, metadata)),
            patch("rich.table.Table", return_value=mock_table),
            patch("rich.console.Console") as mock_console,
            patch("voicepad.cli.config._show_config_hint"),
        ):
            show_config()
        self.assertTrue(mock_table.add_row.called)
        self.assertTrue(mock_console.return_value.print.called)

    def test_list_input_devices_no_devices_raises(self) -> None:
        with patch("voicepad.cli.config._get_input_devices", return_value=[]), self.assertRaises(typer.Exit):
            list_input_devices()

    def test_list_input_devices_with_configured_device(self) -> None:
        cfg = SimpleNamespace(input_device_index=1)
        devices = [AudioDevice(index=1, name="Mic", channels=1, sample_rate=16000)]
        with (
            patch("voicepad.cli.config._get_input_devices", return_value=devices),
            patch("voicepad.cli.config.get_config", return_value=cfg),
            patch("voicepad.cli.config.typer.echo") as mock_echo,
            patch("voicepad.cli.config._show_config_hint"),
        ):
            list_input_devices()
        self.assertTrue(mock_echo.called)

    def test_show_config_hint_branches(self) -> None:
        defaults_meta = SimpleNamespace(per_field={"x": SimpleNamespace(source_path=None, source="defaults")})
        with (
            patch("voicepad.cli.config.get_config_with_metadata", return_value=(SimpleNamespace(), defaults_meta)),
            patch("voicepad.cli.config.typer.echo") as mock_echo,
        ):
            _show_config_hint()
        self.assertTrue(any("using defaults" in str(c.args[0]).lower() for c in mock_echo.call_args_list if c.args))

        file_meta = SimpleNamespace(
            per_field={"x": SimpleNamespace(source_path="D:/voicepad/voicepad.yaml", source="yaml")}
        )
        with (
            patch("voicepad.cli.config.get_config_with_metadata", return_value=(SimpleNamespace(), file_meta)),
            patch("voicepad.cli.config.typer.echo") as mock_echo,
        ):
            _show_config_hint()
        self.assertTrue(any("config file:" in str(c.args[0]).lower() for c in mock_echo.call_args_list if c.args))

    def test_system_info(self) -> None:
        ram = SimpleNamespace(total_gb=16.0, available_gb=8.0)
        cpu = SimpleNamespace(count=8, model_name="CPU")
        gpu = SimpleNamespace(
            nvidia_smi=SimpleNamespace(success=True),
            ctranslate2_cuda=SimpleNamespace(success=True, cuda_device_count=1),
            faster_whisper_gpu=SimpleNamespace(success=True),
        )
        with (
            patch("voicepad_core.get_ram_info", return_value=ram),
            patch("voicepad_core.get_cpu_info", return_value=cpu),
            patch("voicepad_core.gpu_diagnostics", return_value=gpu),
            patch("voicepad.cli.config.typer.echo") as mock_echo,
        ):
            system_info()
        self.assertTrue(mock_echo.called)

    def test_recommend_model(self) -> None:
        recommendation = SimpleNamespace(
            recommended_model="small",
            recommended_device="cpu",
            recommended_compute_type="int8",
            reason="ok",
            alternative_models=["tiny"],
        )
        ram = RAMInfo(total_gb=8.0, available_gb=4.0)
        cpu = CPUInfo(count=4, model_name="x")
        gpu = GPUDiagnosticsReport(
            nvidia_smi=NvidiaCheckResult(success=False, output="no"),
            ctranslate2_cuda=CTranslate2Result(success=False, cuda_device_count=0),
            faster_whisper_gpu=WhisperGPUResult(success=False, message="no"),
        )
        with (
            patch("voicepad_core.get_ram_info", return_value=ram),
            patch("voicepad_core.get_cpu_info", return_value=cpu),
            patch("voicepad_core.gpu_diagnostics", return_value=gpu),
            patch("voicepad_core.get_available_models", return_value=["tiny", "small"]),
            patch("voicepad_core.get_model_recommendation", return_value=recommendation),
            patch("voicepad.cli.config.typer.echo") as mock_echo,
        ):
            recommend_model()
        self.assertTrue(mock_echo.called)

    def test_transcription_config_match_and_mismatch(self) -> None:
        recommendation = SimpleNamespace(
            recommended_model="small",
            recommended_device="cpu",
            recommended_compute_type="int8",
            reason="ok",
        )
        cfg_match = SimpleNamespace(
            transcription_model="small",
            transcription_device="cpu",
            transcription_compute_type="int8",
        )
        ram = RAMInfo(total_gb=8.0, available_gb=4.0)
        cpu = CPUInfo(count=4, model_name="x")
        gpu = GPUDiagnosticsReport(
            nvidia_smi=NvidiaCheckResult(success=False, output="no"),
            ctranslate2_cuda=CTranslate2Result(success=False, cuda_device_count=0),
            faster_whisper_gpu=WhisperGPUResult(success=False, message="no"),
        )
        with (
            patch("voicepad_core.get_config", return_value=cfg_match),
            patch("voicepad_core.get_ram_info", return_value=ram),
            patch("voicepad_core.get_cpu_info", return_value=cpu),
            patch("voicepad_core.gpu_diagnostics", return_value=gpu),
            patch("voicepad_core.get_available_models", return_value=["tiny", "small"]),
            patch("voicepad_core.get_model_recommendation", return_value=recommendation),
            patch("voicepad.cli.config.typer.secho") as mock_secho,
            patch("voicepad.cli.config._show_config_hint"),
        ):
            transcription_config()
        self.assertTrue(any("matches" in str(c.args[0]).lower() for c in mock_secho.call_args_list if c.args))

        cfg_mismatch = SimpleNamespace(
            transcription_model="tiny",
            transcription_device="cpu",
            transcription_compute_type="int8",
        )
        with (
            patch("voicepad_core.get_config", return_value=cfg_mismatch),
            patch("voicepad_core.get_ram_info", return_value=ram),
            patch("voicepad_core.get_cpu_info", return_value=cpu),
            patch("voicepad_core.gpu_diagnostics", return_value=gpu),
            patch("voicepad_core.get_available_models", return_value=["tiny", "small"]),
            patch("voicepad_core.get_model_recommendation", return_value=recommendation),
            patch("voicepad.cli.config.typer.secho") as mock_secho,
            patch("voicepad.cli.config._show_config_hint"),
        ):
            transcription_config()
        self.assertTrue(any("consider updating" in str(c.args[0]).lower() for c in mock_secho.call_args_list if c.args))

    def test_list_models(self) -> None:
        models = ["tiny", "tiny.en", "base", "small", "medium", "large-v3", "turbo", "distil-large-v3"]
        with (
            patch("voicepad_core.get_available_models", return_value=models),
            patch("voicepad.cli.config.typer.echo") as mock_echo,
        ):
            list_models()
        self.assertTrue(mock_echo.called)


if __name__ == "__main__":
    unittest.main()
