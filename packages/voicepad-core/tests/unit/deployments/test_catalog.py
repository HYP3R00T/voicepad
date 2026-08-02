import pytest
from voicepad_core.deployments import (
    PARAKEET_V3_CUDA,
    PARAKEET_V3_MANIFEST,
    SILERO_VAD_ONNX_EXTRACTION,
    SILERO_VAD_WHEEL_MANIFEST,
    ArtifactFile,
    HttpSource,
    HuggingFaceSource,
    Precision,
    get_deployment,
    get_manifest,
)


def test_official_parakeet_manifest_is_pinned() -> None:
    manifest = PARAKEET_V3_MANIFEST

    assert isinstance(manifest.source, HuggingFaceSource)
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


def test_official_silero_artifacts_are_pinned() -> None:
    manifest = SILERO_VAD_WHEEL_MANIFEST
    extraction = SILERO_VAD_ONNX_EXTRACTION

    assert isinstance(manifest.source, HttpSource)
    assert manifest.total_size == 9_146_242
    assert manifest.files[0].sha256 == "09de93c4d874bb19c53e62a47dd38be5f163cedad2b5599583231f2a84ef79cb"
    assert extraction.entry_path == "silero_vad/data/silero_vad.onnx"
    assert extraction.size == 2_327_524
    assert extraction.compressed_size == 1_946_042
    assert extraction.sha256 == "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


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
