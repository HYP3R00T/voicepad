from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArchiveFormat = Literal["zip", "tar"]


@dataclass(frozen=True, slots=True)
class HuggingFaceArtifact:
    """A pinned or floating Hugging Face repository snapshot."""

    repo_id: str
    revision: str | None = None
    allow_patterns: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class DirectUrlArtifact:
    """A file or archive fetched from a direct URL."""

    url: str
    sha256: str | None = None
    archive: ArchiveFormat | None = None
    filename: str | None = None
    root: str | None = None


@dataclass(frozen=True, slots=True)
class LocalArtifact:
    """A model artifact already present on the local filesystem."""

    path: Path


ArtifactSource = HuggingFaceArtifact | DirectUrlArtifact | LocalArtifact

_DEFAULT_CTRANSLATE2_FILES = ("model.bin", "tokenizer.json", "config.json")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """VoicePad model identity, runtime binding, and artifact declaration."""

    id: str
    artifact_source: ArtifactSource
    distil: bool = False
    description: str | None = None
    family: str = "whisper"
    backend_id: str = "faster-whisper"
    artifact_format: str = "ctranslate2"
    quantization: str | None = None
    required_files: tuple[str, ...] = _DEFAULT_CTRANSLATE2_FILES


def _whisper(model_id: str, repo_id: str, *, distil: bool = False) -> ModelSpec:
    return ModelSpec(model_id, HuggingFaceArtifact(repo_id), distil=distil)


_PARAKEET_FILES = (
    "config.json",
    "decoder_joint-model.fp16.onnx",
    "encoder-model.fp16.onnx",
    "nemo128.onnx",
    "vocab.txt",
)


_BUILTIN_MODELS: tuple[ModelSpec, ...] = (
    _whisper("turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    _whisper("small", "Systran/faster-whisper-small"),
    _whisper("distil-large-v3.5", "distil-whisper/distil-large-v3.5-ct2", distil=True),
    ModelSpec(
        "parakeet-tdt-0.6b-v3",
        HuggingFaceArtifact(
            "ysdede/parakeet-tdt-0.6b-v3-onnx",
            revision="f88260fa0777fe0868dda6df85d1a98f012a4a7a",
            allow_patterns=_PARAKEET_FILES,
        ),
        family="parakeet",
        backend_id="parakeet-onnx",
        artifact_format="onnx",
        quantization="fp16",
        description="FP16 ONNX conversion of NVIDIA Parakeet TDT 0.6B v3.",
        required_files=_PARAKEET_FILES,
    ),
)

_model_registry: dict[str, ModelSpec] = {spec.id: spec for spec in _BUILTIN_MODELS}

_MODEL_UI: dict[str, dict[str, str | bool]] = {
    "turbo": {
        "label": "Recommended · Turbo",
        "hint": "~800 MB · best default for most users · fast · multilingual · ~3 GB VRAM",
        "basic": True,
    },
    "small": {
        "label": "Lightweight · Small",
        "hint": "~250 MB · lighter on weaker hardware · good accuracy · multilingual · ~1 GB VRAM",
        "basic": True,
    },
    "distil-large-v3.5": {
        "label": "English · Distil Large v3.5",
        "hint": "~760 MB · fast · English only · ~3 GB VRAM",
        "basic": True,
    },
    "parakeet-tdt-0.6b-v3": {
        "label": "NVIDIA · Parakeet v3",
        "hint": "~1.3 GB · ONNX FP16 · multilingual · NVIDIA CUDA required",
        "basic": True,
    },
}


class ModelCompatibilityError(ValueError):
    """Raised when a model snapshot does not match VoicePad's compatibility checks."""


def _validate_spec(spec: ModelSpec) -> None:
    if not spec.id.strip():
        raise ValueError("Model id must not be empty.")
    if not spec.family.strip():
        raise ValueError(f"Model '{spec.id}' must define a family.")
    if not spec.backend_id.strip():
        raise ValueError(f"Model '{spec.id}' must define a backend_id.")
    if not spec.artifact_format.strip():
        raise ValueError(f"Model '{spec.id}' must define an artifact_format.")
    if not spec.required_files:
        raise ValueError(f"Model '{spec.id}' must declare required_files.")
    for required_file in spec.required_files:
        relative_path = Path(required_file)
        if not required_file.strip() or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Model '{spec.id}' has an invalid required file path: {required_file!r}.")

    artifact_source = spec.artifact_source
    if isinstance(artifact_source, HuggingFaceArtifact):
        if not artifact_source.repo_id.strip():
            raise ValueError(f"Model '{spec.id}' must define a Hugging Face repo_id.")
        if artifact_source.allow_patterns is not None and (
            not artifact_source.allow_patterns or any(not pattern.strip() for pattern in artifact_source.allow_patterns)
        ):
            raise ValueError(f"Model '{spec.id}' has invalid Hugging Face allow_patterns.")
    elif isinstance(artifact_source, DirectUrlArtifact):
        if not artifact_source.url.strip():
            raise ValueError(f"Model '{spec.id}' must define a direct artifact URL.")
        if artifact_source.archive not in (None, "zip", "tar"):
            raise ValueError(f"Model '{spec.id}' has an unsupported archive format.")
        if artifact_source.root is not None:
            artifact_root = Path(artifact_source.root)
            if not artifact_source.root.strip() or artifact_root.is_absolute() or ".." in artifact_root.parts:
                raise ValueError(f"Model '{spec.id}' has an invalid archive root.")
        if artifact_source.sha256 is not None:
            digest = artifact_source.sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"Model '{spec.id}' has an invalid SHA-256 digest.")
    elif isinstance(artifact_source, LocalArtifact) and not str(artifact_source.path):
        raise ValueError(f"Model '{spec.id}' must define a local artifact path.")


def register_model(spec: ModelSpec, *, overwrite: bool = False) -> None:
    """Register a model in the VoicePad catalog."""
    _validate_spec(spec)
    if spec.id in _model_registry and not overwrite:
        raise ValueError(f"Model '{spec.id}' is already registered.")
    _model_registry[spec.id] = spec


def resolve_model_spec(model_id: str) -> ModelSpec:
    """Return a registered model specification."""
    try:
        return _model_registry[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown transcription model '{model_id}'.") from exc


def list_model_specs() -> tuple[ModelSpec, ...]:
    """Return the current model catalog in registration order."""
    return tuple(_model_registry.values())


def list_model_ids() -> tuple[str, ...]:
    """Return the current model ids in registration order."""
    return tuple(_model_registry)


def list_basic_model_ids() -> tuple[str, ...]:
    """Return the curated beginner-friendly model ids."""
    return tuple(model_id for model_id in list_model_ids() if _MODEL_UI.get(model_id, {}).get("basic") is True)


def get_model_label(model_id: str) -> str:
    """Return a user-facing label for a model id."""
    meta = _MODEL_UI.get(model_id, {})
    label = meta.get("label")
    if isinstance(label, str) and label.strip():
        return label
    return model_id


def get_model_hint(model_id: str) -> str:
    """Return a short user-facing hint for a model id."""
    meta = _MODEL_UI.get(model_id, {})
    hint = meta.get("hint")
    if isinstance(hint, str):
        return hint
    return ""


def list_basic_model_options(current_model: str | None = None) -> tuple[tuple[str, str], ...]:
    """Return curated (label, value) pairs for onboarding and simple settings."""
    ids = list(list_basic_model_ids())
    if current_model and current_model in _model_registry and current_model not in ids:
        ids.append(current_model)

    options: list[tuple[str, str]] = []
    for model_id in ids:
        label = get_model_label(model_id)
        if current_model == model_id and model_id not in list_basic_model_ids():
            label = f"Current Advanced · {model_id}"
        options.append((label, model_id))
    return tuple(options)


def validate_model_artifact(artifact_path: str | Path, model: ModelSpec | str) -> Path:
    """Validate an artifact using its catalogue-declared required files."""
    artifact_dir = Path(artifact_path)
    if not artifact_dir.is_dir():
        raise ModelCompatibilityError(f"Model artifact directory not found: {artifact_dir}")

    if isinstance(model, str):
        model = resolve_model_spec(model)
    missing = [name for name in model.required_files if not (artifact_dir / name).is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise ModelCompatibilityError(f"Model artifact is missing required files: {missing_list}")
    return artifact_dir


__all__ = [
    "ArtifactSource",
    "DirectUrlArtifact",
    "HuggingFaceArtifact",
    "LocalArtifact",
    "ModelCompatibilityError",
    "ModelSpec",
    "get_model_hint",
    "get_model_label",
    "list_basic_model_ids",
    "list_basic_model_options",
    "list_model_ids",
    "list_model_specs",
    "register_model",
    "resolve_model_spec",
    "validate_model_artifact",
]
