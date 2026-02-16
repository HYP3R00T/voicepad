from voicepad_core.config import Config, get_config, get_config_with_metadata
from voicepad_core.diagnostics import (
    CPUInfo,
    CTranslate2Result,
    GPUDiagnosticsReport,
    ModelRecommendation,
    NvidiaCheckResult,
    RAMInfo,
    SystemInfo,
    WhisperGPUResult,
    check_ctranslate2_gpu,
    check_faster_whisper_gpu,
    check_nvidia_smi,
    get_available_models,
    get_cpu_info,
    get_model_recommendation,
    get_ram_info,
    gpu_diagnostics,
)
from voicepad_core.recorder import AudioRecorder, AudioRecorderError

__all__ = [
    # Config
    "Config",
    "get_config",
    "get_config_with_metadata",
    # Audio recording
    "AudioRecorder",
    "AudioRecorderError",
    # System detection
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
