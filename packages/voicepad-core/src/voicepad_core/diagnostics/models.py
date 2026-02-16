from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NvidiaCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Whether nvidia-smi command executed successfully")
    output: str = Field(description="Command output or error message")


class CTranslate2Result(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Whether CUDA devices were detected")
    cuda_device_count: int | None = Field(default=None, description="Number of available CUDA devices")
    error_message: str | None = Field(default=None, description="Error message if detection failed")


class WhisperGPUResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Whether the Whisper model loaded successfully on GPU")
    message: str = Field(description="Status message or error details")


class GPUDiagnosticsReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    nvidia_smi: NvidiaCheckResult = Field(description="NVIDIA driver check via nvidia-smi")
    ctranslate2_cuda: CTranslate2Result = Field(description="CTranslate2 CUDA device detection")
    faster_whisper_gpu: WhisperGPUResult = Field(description="Faster Whisper GPU model loading capability")


class RAMInfo(BaseModel):
    """System RAM information."""

    model_config = ConfigDict(frozen=True)

    total_gb: float = Field(description="Total system RAM in GB")
    available_gb: float = Field(description="Available system RAM in GB")


class CPUInfo(BaseModel):
    """System CPU information."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(description="Number of CPU cores (logical)")
    model_name: str | None = Field(default=None, description="CPU model name if available")


class SystemInfo(BaseModel):
    """Complete system information."""

    model_config = ConfigDict(frozen=True)

    ram: RAMInfo = Field(description="RAM information")
    cpu: CPUInfo = Field(description="CPU information")
    gpu_diagnostics: GPUDiagnosticsReport = Field(description="GPU diagnostics report")


class ModelRecommendation(BaseModel):
    """Whisper model recommendation based on system capabilities."""

    model_config = ConfigDict(frozen=True)

    recommended_model: str = Field(description="Recommended model name from available_models()")
    recommended_device: str = Field(description="Recommended device (cuda/cpu)")
    recommended_compute_type: str = Field(description="Recommended compute type (float16/int8/float32)")
    reason: str = Field(description="Explanation for this recommendation")
    alternative_models: list[str] = Field(description="Other viable model options")
    all_available_models: list[str] = Field(description="Complete list from available_models()")
