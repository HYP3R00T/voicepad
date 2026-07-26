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


_BUILTIN_MODELS: tuple[ModelSpec, ...] = (
    _whisper("tiny.en", "Systran/faster-whisper-tiny.en"),
    _whisper("tiny", "Systran/faster-whisper-tiny"),
    _whisper("base.en", "Systran/faster-whisper-base.en"),
    _whisper("base", "Systran/faster-whisper-base"),
    _whisper("small.en", "Systran/faster-whisper-small.en"),
    _whisper("small", "Systran/faster-whisper-small"),
    _whisper("medium.en", "Systran/faster-whisper-medium.en"),
    _whisper("medium", "Systran/faster-whisper-medium"),
    _whisper("large-v1", "Systran/faster-whisper-large-v1"),
    _whisper("large-v2", "Systran/faster-whisper-large-v2"),
    _whisper("large-v3", "Systran/faster-whisper-large-v3"),
    _whisper("large", "Systran/faster-whisper-large-v3"),
    _whisper("distil-small.en", "distil-whisper/distil-small.en", distil=True),
    _whisper("distil-medium.en", "distil-whisper/distil-medium.en", distil=True),
    _whisper("distil-large-v2", "distil-whisper/distil-large-v2", distil=True),
    _whisper("distil-large-v3", "distil-whisper/distil-large-v3", distil=True),
    _whisper("distil-large-v3.5", "distil-whisper/distil-large-v3.5", distil=True),
    _whisper("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    _whisper("turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    ModelSpec(
        "parakeet-tdt-0.6b-v3-int8",
        DirectUrlArtifact(
            url="https://blob.handy.computer/parakeet-v3-int8.tar.gz",
            sha256="43d37191602727524a7d8c6da0eef11c4ba24320f5b4730f1a2497befc2efa77",
            archive="tar",
            root="parakeet-tdt-0.6b-v3-int8",
        ),
        family="parakeet",
        backend_id="parakeet-onnx",
        artifact_format="onnx",
        quantization="int8",
        description="Multilingual Parakeet TDT 0.6B v3, INT8 ONNX bundle.",
        required_files=(
            "decoder_joint-model.int8.onnx",
            "encoder-model.int8.onnx",
            "nemo128.onnx",
            "vocab.txt",
        ),
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
        "label": "Faster · Small",
        "hint": "~250 MB · lighter on weaker hardware · good accuracy · multilingual · ~1 GB VRAM",
        "basic": True,
    },
    "large-v3": {
        "label": "Highest Accuracy · Large v3",
        "hint": "~1.5 GB · best accuracy · slower · multilingual · ~5 GB VRAM",
        "basic": True,
    },
    "parakeet-tdt-0.6b-v3-int8": {
        "label": "NVIDIA · Parakeet v3 INT8",
        "hint": "~480 MB · multilingual · optimized for NVIDIA GPU · proper-noun bias",
        "basic": True,
    },
    "tiny.en": {"label": "Tiny English", "hint": "~40 MB · fastest · English only · CPU"},
    "tiny": {"label": "Tiny", "hint": "~40 MB · fastest · multilingual · CPU"},
    "base.en": {"label": "Base English", "hint": "~75 MB · very fast · English only · CPU"},
    "base": {"label": "Base", "hint": "~75 MB · very fast · multilingual · CPU"},
    "small.en": {"label": "Small English", "hint": "~250 MB · fast · English only · ~1 GB VRAM"},
    "medium.en": {"label": "Medium English", "hint": "~770 MB · moderate · English only · ~2 GB VRAM"},
    "medium": {"label": "Medium", "hint": "~770 MB · moderate · multilingual · ~2 GB VRAM"},
    "large-v1": {"label": "Large v1", "hint": "~1.5 GB · slow · multilingual · ~5 GB VRAM"},
    "large-v2": {"label": "Large v2", "hint": "~1.5 GB · slow · multilingual · ~5 GB VRAM"},
    "large": {"label": "Large", "hint": "~1.5 GB · slow · multilingual · ~5 GB VRAM"},
    "large-v3-turbo": {"label": "Turbo Alias", "hint": "~800 MB · same download as turbo"},
    "distil-small.en": {"label": "Distil Small English", "hint": "~135 MB · fast · English only · ~1 GB VRAM"},
    "distil-medium.en": {"label": "Distil Medium English", "hint": "~395 MB · moderate · English only · ~2 GB VRAM"},
    "distil-large-v2": {"label": "Distil Large v2", "hint": "~760 MB · fast · English only · ~3 GB VRAM"},
    "distil-large-v3": {"label": "Distil Large v3", "hint": "~760 MB · fast · English only · ~3 GB VRAM"},
    "distil-large-v3.5": {"label": "Distil Large v3.5", "hint": "~760 MB · fast · English only · ~3 GB VRAM"},
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
