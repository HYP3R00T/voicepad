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

    def test_builtin_parakeet_declares_runtime_and_verified_archive(self) -> None:
        """The Parakeet entry pins its ONNX layout and published archive digest."""
        spec = resolve_model_spec("parakeet-tdt-0.6b-v3-int8")

        assert (
            spec.backend_id,
            spec.artifact_format,
            spec.quantization,
            spec.required_files,
            spec.artifact_source,
        ) == (
            "parakeet-onnx",
            "onnx",
            "int8",
            (
                "decoder_joint-model.int8.onnx",
                "encoder-model.int8.onnx",
                "nemo128.onnx",
                "vocab.txt",
            ),
            DirectUrlArtifact(
                url="https://blob.handy.computer/parakeet-v3-int8.tar.gz",
                sha256="43d37191602727524a7d8c6da0eef11c4ba24320f5b4730f1a2497befc2efa77",
                archive="tar",
                root="parakeet-tdt-0.6b-v3-int8",
            ),
        )

    def test_builtin_parakeet_is_curated_for_basic_model_selection(self) -> None:
        """The supported Parakeet runtime appears in simple model pickers with useful metadata."""
        model_id = "parakeet-tdt-0.6b-v3-int8"

        assert (
            model_id in list_basic_model_ids(),
            get_model_label(model_id),
            get_model_hint(model_id),
        ) == (
            True,
            "NVIDIA · Parakeet v3 INT8",
            "~480 MB · multilingual · optimized for NVIDIA GPU · proper-noun bias",
        )
