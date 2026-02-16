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
