from __future__ import annotations

from .types import (
    ArtifactFile,
    ArtifactManifest,
    DeclaredCapabilities,
    DeploymentDefinition,
    HttpSource,
    HuggingFaceSource,
    Precision,
    ProcessingProfile,
    ResourceProfile,
    TimestampGranularity,
    WheelExtraction,
)

PARAKEET_V3_MANIFEST = ArtifactManifest(
    id="official-parakeet-v3-safetensors",
    source=HuggingFaceSource(
        repository="nvidia/parakeet-tdt-0.6b-v3",
        revision="7c35754d166cca382ad1e53e68b01e7c575f3a1d",
    ),
    license="CC-BY-4.0",
    files=(
        ArtifactFile(
            "model.safetensors",
            2_508_311_120,
            "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
        ),
        ArtifactFile(
            "config.json",
            1_153,
            "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e",
        ),
        ArtifactFile(
            "generation_config.json",
            289,
            "b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0",
        ),
        ArtifactFile(
            "processor_config.json",
            392,
            "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7",
        ),
        ArtifactFile(
            "tokenizer.json",
            1_159_960,
            "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
        ),
        ArtifactFile(
            "tokenizer_config.json",
            290,
            "0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874",
        ),
    ),
)

SILERO_VAD_WHEEL_MANIFEST = ArtifactManifest(
    id="official-silero-vad-6.2.1-wheel",
    source=HttpSource(
        "https://files.pythonhosted.org/packages/0b/2b/48566f29a8b53d856ceb1994f209122749b3fda0a733a07e82047257de7a/"
        "silero_vad-6.2.1-py3-none-any.whl"
    ),
    license="MIT",
    files=(
        ArtifactFile(
            "silero_vad-6.2.1-py3-none-any.whl",
            9_146_242,
            "09de93c4d874bb19c53e62a47dd38be5f163cedad2b5599583231f2a84ef79cb",
        ),
    ),
)

SILERO_VAD_ONNX_EXTRACTION = WheelExtraction(
    id="official-silero-vad-6.2.1-onnx",
    wheel_manifest_id=SILERO_VAD_WHEEL_MANIFEST.id,
    entry_path="silero_vad/data/silero_vad.onnx",
    output_name="silero_vad.onnx",
    size=2_327_524,
    compressed_size=1_946_042,
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
)

PARAKEET_V3_CUDA = DeploymentDefinition(
    id="parakeet-v3.transformers-fp16-cuda",
    model_id="nvidia-parakeet-tdt-0.6b-v3",
    artifact_manifest_id=PARAKEET_V3_MANIFEST.id,
    adapter_id="transformers-parakeet-tdt",
    precision=Precision.FP16,
    capabilities=DeclaredCapabilities(
        native_sample_rate=16_000,
        languages=(
            "bg",
            "hr",
            "cs",
            "da",
            "nl",
            "en",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "it",
            "lv",
            "lt",
            "mt",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "es",
            "sv",
            "ru",
            "uk",
        ),
        timestamps=TimestampGranularity.TOKEN_DURATION,
    ),
    resources=ResourceProfile(
        required_device="cuda",
        platforms=("linux-x86_64",),
        minimum_gpu_memory_bytes=3_900_000_000,
        measured_gpu="NVIDIA GeForce RTX 3050 Laptop GPU",
        measured_peak_memory_bytes=2_040_109_056,
    ),
    processing=ProcessingProfile(
        preferred_chunk_seconds=30,
        maximum_input_seconds=60,
        warmup_seconds=30,
    ),
    recommended=True,
)

MANIFESTS = {
    PARAKEET_V3_MANIFEST.id: PARAKEET_V3_MANIFEST,
    SILERO_VAD_WHEEL_MANIFEST.id: SILERO_VAD_WHEEL_MANIFEST,
}
DEPLOYMENTS = {PARAKEET_V3_CUDA.id: PARAKEET_V3_CUDA}


def get_manifest(manifest_id: str) -> ArtifactManifest:
    try:
        return MANIFESTS[manifest_id]
    except KeyError as error:
        raise KeyError(f"Unknown artifact manifest: {manifest_id}") from error


def get_deployment(deployment_id: str) -> DeploymentDefinition:
    try:
        return DEPLOYMENTS[deployment_id]
    except KeyError as error:
        raise KeyError(f"Unknown deployment: {deployment_id}") from error
