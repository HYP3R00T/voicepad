"""Unit tests for diagnostics modules."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicepad_core.diagnostics.gpu import (
    check_ctranslate2_gpu,
    check_faster_whisper_gpu,
    check_nvidia_smi,
    gpu_diagnostics,
)
from voicepad_core.diagnostics.models import (
    CPUInfo,
    CTranslate2Result,
    GPUDiagnosticsReport,
    NvidiaCheckResult,
    RAMInfo,
    SystemInfo,
    WhisperGPUResult,
)
from voicepad_core.diagnostics.recommendations import (
    categorize_model,
    estimate_vram_gb,
    get_model_recommendation,
)
from voicepad_core.diagnostics.system import get_available_models, get_cpu_info, get_ram_info


class DiagnosticsTests(unittest.TestCase):
    def test_check_nvidia_smi_success(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with patch("voicepad_core.diagnostics.gpu.subprocess.run", return_value=completed):
            result = check_nvidia_smi()
        self.assertTrue(result.success)
        self.assertEqual(result.output, "ok")

    def test_check_nvidia_smi_not_found(self) -> None:
        with patch("voicepad_core.diagnostics.gpu.subprocess.run", side_effect=FileNotFoundError):
            result = check_nvidia_smi()
        self.assertFalse(result.success)
        self.assertIn("not found", result.output)

    def test_check_nvidia_smi_timeout(self) -> None:
        with patch("voicepad_core.diagnostics.gpu.subprocess.run", side_effect=subprocess.TimeoutExpired("x", 5)):
            result = check_nvidia_smi()
        self.assertFalse(result.success)
        self.assertIn("timed out", result.output)

    def test_check_ctranslate2_gpu_paths(self) -> None:
        with patch("voicepad_core.diagnostics.gpu.ctranslate2.get_cuda_device_count", return_value=2):
            ok_result = check_ctranslate2_gpu()
        self.assertTrue(ok_result.success)
        self.assertEqual(ok_result.cuda_device_count, 2)

        with patch("voicepad_core.diagnostics.gpu.ctranslate2.get_cuda_device_count", return_value=0):
            no_gpu = check_ctranslate2_gpu()
        self.assertFalse(no_gpu.success)
        self.assertEqual(no_gpu.cuda_device_count, 0)

        with patch("voicepad_core.diagnostics.gpu.ctranslate2.get_cuda_device_count", side_effect=RuntimeError("boom")):
            error_result = check_ctranslate2_gpu()
        self.assertFalse(error_result.success)
        self.assertIn("boom", error_result.error_message or "")

    def test_check_faster_whisper_gpu_paths(self) -> None:
        with patch("voicepad_core.diagnostics.gpu.WhisperModel"):
            ok_result = check_faster_whisper_gpu()
        self.assertTrue(ok_result.success)

        with patch("voicepad_core.diagnostics.gpu.WhisperModel", side_effect=RuntimeError("gpu init failed")):
            error_result = check_faster_whisper_gpu()
        self.assertFalse(error_result.success)
        self.assertIn("gpu init failed", error_result.message)

    def test_gpu_diagnostics_aggregates_checks(self) -> None:
        nvidia = NvidiaCheckResult(success=True, output="ok")
        ct2 = CTranslate2Result(success=True, cuda_device_count=1)
        whisper = WhisperGPUResult(success=True, message="ok")

        with (
            patch("voicepad_core.diagnostics.gpu.check_nvidia_smi", return_value=nvidia),
            patch("voicepad_core.diagnostics.gpu.check_ctranslate2_gpu", return_value=ct2),
            patch("voicepad_core.diagnostics.gpu.check_faster_whisper_gpu", return_value=whisper),
        ):
            report = gpu_diagnostics()

        self.assertIsInstance(report, GPUDiagnosticsReport)
        self.assertTrue(report.nvidia_smi.success)

    def test_system_info_helpers(self) -> None:
        with patch("voicepad_core.diagnostics.system.available_models", return_value=["tiny", "small"]):
            models = get_available_models()
        self.assertEqual(models, ["tiny", "small"])

        with patch("voicepad_core.diagnostics.system.available_models", side_effect=RuntimeError("x")):
            fallback_models = get_available_models()
        self.assertIn("tiny", fallback_models)

        mem = SimpleNamespace(total=8 * 1024**3, available=4 * 1024**3)
        with patch("voicepad_core.diagnostics.system.psutil.virtual_memory", return_value=mem):
            ram = get_ram_info()
        self.assertIsInstance(ram, RAMInfo)
        self.assertEqual(ram.total_gb, 8.0)

        with patch("voicepad_core.diagnostics.system.psutil.virtual_memory", side_effect=RuntimeError("x")):
            ram_fallback = get_ram_info()
        self.assertEqual(ram_fallback.total_gb, 0.0)

        with (
            patch("voicepad_core.diagnostics.system.psutil.cpu_count", return_value=8),
            patch("voicepad_core.diagnostics.system.platform.processor", return_value="Ryzen"),
        ):
            cpu = get_cpu_info()
        self.assertIsInstance(cpu, CPUInfo)
        self.assertEqual(cpu.count, 8)
        self.assertEqual(cpu.model_name, "Ryzen")

        with patch("voicepad_core.diagnostics.system.psutil.cpu_count", side_effect=RuntimeError("x")):
            cpu_fallback = get_cpu_info()
        self.assertEqual(cpu_fallback.count, 0)

    def test_recommendation_helpers(self) -> None:
        self.assertEqual(categorize_model("large-v3"), "large")
        self.assertEqual(categorize_model("turbo"), "turbo")
        self.assertEqual(categorize_model("distil-large-v3"), "distil")
        self.assertEqual(categorize_model("unknown"), "tiny")

        self.assertEqual(estimate_vram_gb(None), 4.0)
        self.assertEqual(estimate_vram_gb("GPU 0: 8192MiB"), 8.0)
        self.assertEqual(estimate_vram_gb("Memory: 16GB"), 16.0)

    def test_get_model_recommendation_gpu_and_cpu(self) -> None:
        available = ["tiny", "base", "small", "medium", "large-v3", "turbo", "small.en"]

        gpu_system = SystemInfo(
            ram=RAMInfo(total_gb=16.0, available_gb=12.0),
            cpu=CPUInfo(count=8, model_name="x"),
            gpu_diagnostics=GPUDiagnosticsReport(
                nvidia_smi=NvidiaCheckResult(success=True, output="8192MiB"),
                ctranslate2_cuda=CTranslate2Result(success=True, cuda_device_count=1),
                faster_whisper_gpu=WhisperGPUResult(success=True, message="ok"),
            ),
        )
        gpu_rec = get_model_recommendation(gpu_system, available)
        self.assertEqual(gpu_rec.recommended_device, "cuda")

        cpu_system = SystemInfo(
            ram=RAMInfo(total_gb=8.0, available_gb=3.0),
            cpu=CPUInfo(count=4, model_name="x"),
            gpu_diagnostics=GPUDiagnosticsReport(
                nvidia_smi=NvidiaCheckResult(success=False, output="no"),
                ctranslate2_cuda=CTranslate2Result(success=False, cuda_device_count=0),
                faster_whisper_gpu=WhisperGPUResult(success=False, message="no"),
            ),
        )
        cpu_rec = get_model_recommendation(cpu_system, available)
        self.assertEqual(cpu_rec.recommended_device, "cpu")


if __name__ == "__main__":
    unittest.main()
