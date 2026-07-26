from __future__ import annotations

from pathlib import Path

import pytest
from voicepad_core.models import (
    ArtifactSource,
    DirectUrlArtifact,
    HuggingFaceArtifact,
    LocalArtifact,
    ModelSpec,
    get_model_hint,
    get_model_label,
    list_basic_model_ids,
    register_model,
    resolve_model_spec,
    validate_model_artifact,
)


class TestModelSpec:
    def test_whisper_defaults_describe_existing_artifacts(self) -> None:
        spec = ModelSpec("test-defaults", HuggingFaceArtifact("owner/model"))

        assert (spec.family, spec.backend_id, spec.artifact_format) == (
            "whisper",
            "faster-whisper",
            "ctranslate2",
        )

    @pytest.mark.parametrize(
        ("field", "message"),
        (
            ("family", "family"),
            ("backend_id", "backend_id"),
            ("artifact_format", "artifact_format"),
        ),
    )
    def test_register_model_rejects_missing_runtime_identity(self, field: str, message: str) -> None:
        """A registered model must identify its family, backend, and artifact format."""
        values = {
            "family": "whisper",
            "backend_id": "faster-whisper",
            "artifact_format": "ctranslate2",
        }
        values[field] = ""
        spec = ModelSpec(
            "invalid-runtime",
            HuggingFaceArtifact("owner/model"),
            family=values["family"],
            backend_id=values["backend_id"],
            artifact_format=values["artifact_format"],
        )

        with pytest.raises(ValueError, match=message):
            register_model(spec, overwrite=True)

    def test_registered_model_retains_runtime_identity(self) -> None:
        """Model resolution returns the backend metadata supplied at registration."""
        model_id = "test-parakeet-runtime"
        register_model(
            ModelSpec(
                model_id,
                HuggingFaceArtifact("owner/parakeet"),
                family="parakeet",
                backend_id="transcribe-cpp",
                artifact_format="gguf",
                quantization="Q8_0",
            ),
            overwrite=True,
        )

        spec = resolve_model_spec(model_id)

        assert (spec.family, spec.backend_id, spec.artifact_format, spec.quantization) == (
            "parakeet",
            "transcribe-cpp",
            "gguf",
            "Q8_0",
        )

    @pytest.mark.parametrize(
        "source",
        (
            DirectUrlArtifact("https://example.test/model.gguf", filename="model.gguf"),
            LocalArtifact(Path("models/model.gguf")),
        ),
    )
    def test_non_hugging_face_sources_do_not_require_repo_id(self, source: ArtifactSource) -> None:
        """Direct and local artifacts are independent of Hugging Face identity."""
        spec = ModelSpec(
            "portable",
            backend_id="native",
            artifact_format="gguf",
            artifact_source=source,
            required_files=("model.gguf",),
        )

        register_model(spec, overwrite=True)

        assert resolve_model_spec("portable").artifact_source == source

    def test_snapshot_validation_uses_declared_layout(self, tmp_path: Path) -> None:
        """A backend-specific artifact is validated without Whisper file assumptions."""
        (tmp_path / "parakeet-q8.gguf").write_bytes(b"gguf")
        spec = ModelSpec(
            "parakeet-q8",
            backend_id="native",
            artifact_format="gguf",
            artifact_source=LocalArtifact(tmp_path),
            required_files=("parakeet-q8.gguf",),
        )

        validated = validate_model_artifact(tmp_path, spec)

        assert validated == tmp_path

    def test_builtin_parakeet_uses_pinned_fp16_onnx_artifact(self) -> None:
        """The Parakeet entry selects only the CUDA-oriented FP16 ONNX files."""
        spec = resolve_model_spec("parakeet-tdt-0.6b-v3")

        assert (
            spec.backend_id,
            spec.artifact_format,
            spec.quantization,
            spec.required_files,
            spec.artifact_source,
        ) == (
            "parakeet-onnx",
            "onnx",
            "fp16",
            (
                "config.json",
                "decoder_joint-model.fp16.onnx",
                "encoder-model.fp16.onnx",
                "nemo128.onnx",
                "vocab.txt",
            ),
            HuggingFaceArtifact(
                "ysdede/parakeet-tdt-0.6b-v3-onnx",
                revision="f88260fa0777fe0868dda6df85d1a98f012a4a7a",
                allow_patterns=(
                    "config.json",
                    "decoder_joint-model.fp16.onnx",
                    "encoder-model.fp16.onnx",
                    "nemo128.onnx",
                    "vocab.txt",
                ),
            ),
        )

    def test_builtin_distil_model_uses_ctranslate2_artifact(self) -> None:
        """The distilled model artifact matches the faster-whisper backend."""
        spec = resolve_model_spec("distil-large-v3.5")

        assert spec.artifact_source == HuggingFaceArtifact("distil-whisper/distil-large-v3.5-ct2")

    def test_builtin_parakeet_is_curated_for_basic_model_selection(self) -> None:
        """The supported Parakeet runtime appears in simple model pickers with useful metadata."""
        model_id = "parakeet-tdt-0.6b-v3"

        assert (
            model_id in list_basic_model_ids(),
            get_model_label(model_id),
            get_model_hint(model_id),
        ) == (
            True,
            "NVIDIA · Parakeet v3",
            "~1.3 GB · ONNX FP16 · multilingual · NVIDIA CUDA required",
        )
