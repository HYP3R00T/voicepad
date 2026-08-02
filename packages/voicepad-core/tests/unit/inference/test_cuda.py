from types import SimpleNamespace

import pytest
from voicepad_core.deployments import PARAKEET_V3_CUDA
from voicepad_core.inference import CudaAdmissionError, UnsupportedPlatformError, admit_cuda_device, cuda


def configure_admissible_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    properties = SimpleNamespace(
        name="NVIDIA Test GPU",
        total_memory=3_953_393_664,
        major=8,
        minor=6,
        uuid="test-device-uuid",
    )
    monkeypatch.setattr(cuda.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cuda.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cuda.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cuda.torch.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.setattr(cuda.torch.cuda, "mem_get_info", lambda index: (3_500_000_000, properties.total_memory))


def test_admit_cuda_device_returns_stable_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_admissible_cuda(monkeypatch)

    device = admit_cuda_device(PARAKEET_V3_CUDA)

    assert device.stable_id == "GPU-test-device-uuid"
    assert device.torch_device == "cuda:0"
    assert device.compute_capability == (8, 6)


def test_admission_rejects_non_linux_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cuda.platform, "machine", lambda: "AMD64")

    with pytest.raises(UnsupportedPlatformError, match="supports"):
        admit_cuda_device(PARAKEET_V3_CUDA)


def test_admission_never_falls_back_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cuda.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cuda.torch.cuda, "is_available", lambda: False)

    with pytest.raises(CudaAdmissionError, match="requires a supported NVIDIA GPU"):
        admit_cuda_device(PARAKEET_V3_CUDA)


def test_admission_rejects_insufficient_free_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_admissible_cuda(monkeypatch)
    monkeypatch.setattr(cuda.torch.cuda, "mem_get_info", lambda index: (1_000_000_000, 3_953_393_664))

    with pytest.raises(CudaAdmissionError, match="Insufficient free GPU memory"):
        admit_cuda_device(PARAKEET_V3_CUDA)
