from __future__ import annotations

import platform
from dataclasses import dataclass

import torch

from voicepad_core.deployments import DeploymentDefinition, Precision

from .types import CudaAdmissionError, UnsupportedPlatformError

ADMISSION_HEADROOM_BYTES = 512 * 1024**2


@dataclass(frozen=True, slots=True)
class CudaDevice:
    index: int
    stable_id: str
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    compute_capability: tuple[int, int]

    @property
    def torch_device(self) -> str:
        return f"cuda:{self.index}"


def admit_cuda_device(deployment: DeploymentDefinition, index: int = 0) -> CudaDevice:
    platform_id = f"{platform.system().lower()}-{platform.machine().lower()}"
    if platform_id not in deployment.resources.platforms:
        raise UnsupportedPlatformError(
            f"Deployment '{deployment.id}' supports {deployment.resources.platforms}, not '{platform_id}'."
        )
    if deployment.resources.required_device != "cuda" or deployment.precision is not Precision.FP16:
        raise CudaAdmissionError(f"Deployment '{deployment.id}' is not a CUDA FP16 deployment.")
    if not torch.cuda.is_available():
        raise CudaAdmissionError("CUDA is unavailable. VoicePad requires a supported NVIDIA GPU and driver.")
    if index < 0 or index >= torch.cuda.device_count():
        raise CudaAdmissionError(f"CUDA device index is unavailable: {index}")

    properties = torch.cuda.get_device_properties(index)
    total_memory = int(properties.total_memory)
    if total_memory < deployment.resources.minimum_gpu_memory_bytes:
        raise CudaAdmissionError(
            f"GPU memory is below the deployment requirement: available={total_memory} "
            f"required={deployment.resources.minimum_gpu_memory_bytes} bytes."
        )

    capability = (int(properties.major), int(properties.minor))
    if capability < (5, 3):
        raise CudaAdmissionError(f"CUDA device does not support required FP16 operations: capability={capability}.")

    try:
        free_memory, runtime_total = torch.cuda.mem_get_info(index)
    except Exception as error:
        raise CudaAdmissionError(f"Could not inspect free CUDA memory: {error}") from error
    required_free = deployment.resources.measured_peak_memory_bytes + ADMISSION_HEADROOM_BYTES
    if int(free_memory) < required_free:
        raise CudaAdmissionError(f"Insufficient free GPU memory: free={free_memory} required={required_free} bytes.")
    if int(runtime_total) != total_memory:
        raise CudaAdmissionError("CUDA memory identity changed during admission.")

    device_uuid = str(properties.uuid)
    if not device_uuid:
        raise CudaAdmissionError("CUDA did not expose a stable NVIDIA device UUID.")
    return CudaDevice(
        index=index,
        stable_id=f"GPU-{device_uuid}",
        name=str(properties.name),
        total_memory_bytes=total_memory,
        free_memory_bytes=int(free_memory),
        compute_capability=capability,
    )
