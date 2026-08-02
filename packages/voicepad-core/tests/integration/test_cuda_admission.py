import pytest
from voicepad_core.deployments import PARAKEET_V3_CUDA
from voicepad_core.inference import admit_cuda_device


@pytest.mark.gpu
def test_primary_nvidia_device_passes_deployment_admission() -> None:
    device = admit_cuda_device(PARAKEET_V3_CUDA)

    assert device.name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert device.total_memory_bytes >= PARAKEET_V3_CUDA.resources.minimum_gpu_memory_bytes
    assert device.compute_capability >= (5, 3)
