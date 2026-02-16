from __future__ import annotations

import subprocess

import ctranslate2
from faster_whisper import WhisperModel

from voicepad_core.diagnostics.models import (
    CTranslate2Result,
    GPUDiagnosticsReport,
    NvidiaCheckResult,
    WhisperGPUResult,
)


def check_nvidia_smi() -> NvidiaCheckResult:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return NvidiaCheckResult(success=True, output=result.stdout)
        return NvidiaCheckResult(success=False, output=result.stderr)
    except FileNotFoundError:
        return NvidiaCheckResult(success=False, output="nvidia-smi command not found in PATH")
    except subprocess.TimeoutExpired:
        return NvidiaCheckResult(success=False, output="nvidia-smi command timed out after 5 seconds")


def check_ctranslate2_gpu() -> CTranslate2Result:
    try:
        device_count = ctranslate2.get_cuda_device_count()  # type: ignore[attr-defined]
        if device_count > 0:
            return CTranslate2Result(success=True, cuda_device_count=device_count)
        return CTranslate2Result(
            success=False,
            cuda_device_count=0,
            error_message="No CUDA devices detected by CTranslate2",
        )
    except Exception as e:
        return CTranslate2Result(success=False, cuda_device_count=None, error_message=str(e))


def check_faster_whisper_gpu() -> WhisperGPUResult:
    try:
        WhisperModel("tiny", device="cuda", compute_type="float16")
        return WhisperGPUResult(success=True, message="Model loaded successfully on GPU")
    except Exception as e:
        return WhisperGPUResult(success=False, message=str(e))


def gpu_diagnostics() -> GPUDiagnosticsReport:
    nvidia_result = check_nvidia_smi()
    ctranslate2_result = check_ctranslate2_gpu()
    whisper_result = check_faster_whisper_gpu()

    return GPUDiagnosticsReport(
        nvidia_smi=nvidia_result,
        ctranslate2_cuda=ctranslate2_result,
        faster_whisper_gpu=whisper_result,
    )
