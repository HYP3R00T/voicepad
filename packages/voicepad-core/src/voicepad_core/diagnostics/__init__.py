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
    ModelRecommendation,
    NvidiaCheckResult,
    RAMInfo,
    SystemInfo,
    WhisperGPUResult,
)
from voicepad_core.diagnostics.recommendations import get_model_recommendation
from voicepad_core.diagnostics.system import get_available_models, get_cpu_info, get_ram_info

__all__ = [
    # System detection functions
    "get_ram_info",
    "get_cpu_info",
    "get_available_models",
    "get_model_recommendation",
    # GPU diagnostics functions
    "check_nvidia_smi",
    "check_ctranslate2_gpu",
    "check_faster_whisper_gpu",
    "gpu_diagnostics",
    # Diagnostic models
    "NvidiaCheckResult",
    "CTranslate2Result",
    "WhisperGPUResult",
    "GPUDiagnosticsReport",
    # System models
    "RAMInfo",
    "CPUInfo",
    "SystemInfo",
    "ModelRecommendation",
]
