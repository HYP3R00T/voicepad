import pytest
from voicepad_core.deployments import (
    PARAKEET_V3_CUDA,
    PARAKEET_V3_MANIFEST,
    ArtifactFile,
    Precision,
    get_deployment,
    get_manifest,
)


def test_official_parakeet_manifest_is_pinned() -> None:
    manifest = PARAKEET_V3_MANIFEST

    assert manifest.source.repository == "nvidia/parakeet-tdt-0.6b-v3"
    assert manifest.source.revision == "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
    assert manifest.total_size == 2_509_473_204
    assert {artifact.path for artifact in manifest.files} == {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }


def test_initial_deployment_requires_linux_cuda_fp16() -> None:
    deployment = PARAKEET_V3_CUDA

    assert deployment.precision is Precision.FP16
    assert deployment.resources.required_device == "cuda"
    assert deployment.resources.platforms == ("linux-x86_64",)
    assert deployment.resources.minimum_gpu_memory_bytes == 3_900_000_000
    assert deployment.processing.maximum_input_seconds == 60
    assert deployment.capabilities.native_sample_rate == 16_000
    assert deployment.capabilities.native_streaming is False
    assert deployment.capabilities.accepts_language_hint is False


def test_catalogue_lookups_use_stable_ids() -> None:
    assert get_manifest(PARAKEET_V3_MANIFEST.id) is PARAKEET_V3_MANIFEST
    assert get_deployment(PARAKEET_V3_CUDA.id) is PARAKEET_V3_CUDA

    with pytest.raises(KeyError, match="Unknown deployment"):
        get_deployment("parakeet")


def test_artifact_file_rejects_unsafe_paths() -> None:
    with pytest.raises(ValueError, match="normalized relative"):
        ArtifactFile("../model.safetensors", 1, "0" * 64)
