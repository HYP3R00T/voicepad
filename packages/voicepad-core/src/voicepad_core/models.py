from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """VoicePad-owned model registry entry."""

    id: str
    repo_id: str
    revision: str | None = None
    distil: bool = False
    source: str = "official"
    description: str | None = None


_BUILTIN_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("tiny.en", "Systran/faster-whisper-tiny.en"),
    ModelSpec("tiny", "Systran/faster-whisper-tiny"),
    ModelSpec("base.en", "Systran/faster-whisper-base.en"),
    ModelSpec("base", "Systran/faster-whisper-base"),
    ModelSpec("small.en", "Systran/faster-whisper-small.en"),
    ModelSpec("small", "Systran/faster-whisper-small"),
    ModelSpec("medium.en", "Systran/faster-whisper-medium.en"),
    ModelSpec("medium", "Systran/faster-whisper-medium"),
    ModelSpec("large-v1", "Systran/faster-whisper-large-v1"),
    ModelSpec("large-v2", "Systran/faster-whisper-large-v2"),
    ModelSpec("large-v3", "Systran/faster-whisper-large-v3"),
    ModelSpec("large", "Systran/faster-whisper-large-v3"),
    ModelSpec("distil-small.en", "distil-whisper/distil-small.en", distil=True),
    ModelSpec("distil-medium.en", "distil-whisper/distil-medium.en", distil=True),
    ModelSpec("distil-large-v2", "distil-whisper/distil-large-v2", distil=True),
    ModelSpec("distil-large-v3", "distil-whisper/distil-large-v3", distil=True),
    ModelSpec("distil-large-v3.5", "distil-whisper/distil-large-v3.5", distil=True),
    ModelSpec("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    ModelSpec("turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
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


class _ModelIdView:
    """Live sequence view so callers importing the symbol see registry updates."""

    def __iter__(self) -> Iterator[str]:
        return iter(list_model_ids())

    def __len__(self) -> int:
        return len(_model_registry)

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        items = list_model_ids()
        return items[index]

    def __contains__(self, value: object) -> bool:
        return isinstance(value, str) and value in _model_registry

    def __repr__(self) -> str:
        return repr(tuple(self))


def _validate_spec(spec: ModelSpec) -> None:
    if not spec.id.strip():
        raise ValueError("Model id must not be empty.")
    if not spec.repo_id.strip():
        raise ValueError(f"Model '{spec.id}' must define a repo_id.")


def register_model(spec: ModelSpec, *, overwrite: bool = False) -> None:
    """Register a model in the VoicePad catalog."""
    _validate_spec(spec)
    if spec.id in _model_registry and not overwrite:
        raise ValueError(f"Model '{spec.id}' is already registered.")
    _model_registry[spec.id] = spec


def register_models(specs: Iterable[ModelSpec], *, overwrite: bool = False) -> None:
    """Register multiple models in the VoicePad catalog."""
    for spec in specs:
        register_model(spec, overwrite=overwrite)


def get_model_spec(model_id: str) -> ModelSpec | None:
    """Return the registered model spec for an id, if present."""
    return _model_registry.get(model_id)


def resolve_model_spec(model_id: str) -> ModelSpec:
    """Resolve a model id to a registered spec or a compatible fallback mapping."""
    spec = get_model_spec(model_id)
    if spec is not None:
        return spec
    return ModelSpec(model_id, f"Systran/faster-whisper-{model_id}", source="fallback")


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


def is_distil_model(model_id: str) -> bool:
    """Return True when the registered model is a distil Whisper variant."""
    return resolve_model_spec(model_id).distil


def validate_model_snapshot(snapshot_path: str | Path) -> Path:
    """Validate that a downloaded snapshot looks compatible with this pipeline."""
    snapshot_dir = Path(snapshot_path)
    if not snapshot_dir.is_dir():
        raise ModelCompatibilityError(f"Model snapshot directory not found: {snapshot_dir}")

    required_files = ("model.bin", "tokenizer.json", "config.json")
    missing = [name for name in required_files if not (snapshot_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise ModelCompatibilityError(f"Model snapshot is missing required files: {missing_list}")

    config_path = snapshot_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelCompatibilityError(f"Model config is not valid JSON: {config_path}") from exc

    model_type = str(config.get("model_type", "")).strip().lower()
    if "whisper" in model_type:
        return snapshot_dir

    whisper_markers = ("alignment_heads", "lang_ids", "suppress_ids", "suppress_ids_begin")
    if any(marker in config for marker in whisper_markers):
        return snapshot_dir

    if "whisper" not in model_type:
        raise ModelCompatibilityError(
            f"Model config at {config_path} is not Whisper-compatible (model_type={model_type!r})."
        )

    return snapshot_dir


VALID_TRANSCRIPTION_MODELS: _ModelIdView = _ModelIdView()

__all__ = [
    "ModelCompatibilityError",
    "ModelSpec",
    "VALID_TRANSCRIPTION_MODELS",
    "get_model_hint",
    "get_model_label",
    "get_model_spec",
    "is_distil_model",
    "list_basic_model_ids",
    "list_basic_model_options",
    "list_model_ids",
    "list_model_specs",
    "register_model",
    "register_models",
    "resolve_model_spec",
    "validate_model_snapshot",
]
