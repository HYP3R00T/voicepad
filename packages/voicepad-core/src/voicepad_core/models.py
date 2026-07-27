from dataclasses import dataclass
from pathlib import Path

_WHISPER_FILES = ("model.bin", "tokenizer.json", "config.json")
_PARAKEET_FILES = (
    "decoder.int8.onnx",
    "encoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)


@dataclass(frozen=True, slots=True)
class Model:
    """One supported model and the backend that can run it."""

    id: str
    repo: str
    backend: str
    files: tuple[str, ...]
    label: str
    hint: str
    revision: str | None = None
    precision: str | None = None
    accepts_prompt: bool = True


MODELS: dict[str, Model] = {
    model.id: model
    for model in (
        Model(
            id="turbo",
            repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            backend="faster-whisper",
            files=_WHISPER_FILES,
            label="Recommended · Turbo",
            hint="~800 MB · best default for most users · fast · multilingual · ~3 GB VRAM",
        ),
        Model(
            id="small",
            repo="Systran/faster-whisper-small",
            backend="faster-whisper",
            files=_WHISPER_FILES,
            label="Lightweight · Small",
            hint="~250 MB · good accuracy · multilingual · ~1 GB VRAM",
        ),
        Model(
            id="distil-large-v3.5",
            repo="distil-whisper/distil-large-v3.5-ct2",
            backend="faster-whisper",
            files=_WHISPER_FILES,
            label="English · Distil Large v3.5",
            hint="~760 MB · fast · English only · ~3 GB VRAM",
            accepts_prompt=False,
        ),
        Model(
            id="parakeet-tdt-0.6b-v3",
            repo="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            backend="sherpa-onnx",
            files=_PARAKEET_FILES,
            revision="2bda32ec70b097a55adaa07d9a7173915b43cc78",
            precision="int8",
            label="NVIDIA · Parakeet v3",
            hint="~670 MB · Sherpa int8 · multilingual · NVIDIA CUDA",
        ),
    )
}


class ModelCompatibilityError(ValueError):
    """Raised when a downloaded model is incomplete."""


def get_model(model_id: str) -> Model:
    """Return a supported model."""
    try:
        return MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown transcription model '{model_id}'.") from exc


def model_options() -> tuple[tuple[str, str], ...]:
    """Return display labels and model identifiers for the UI."""
    return tuple((model.label, model.id) for model in MODELS.values())


def validate_model(path: str | Path, model: Model) -> Path:
    """Return the model directory when every required file exists."""
    directory = Path(path)
    missing = [name for name in model.files if not (directory / name).is_file()]
    if not directory.is_dir() or missing:
        detail = ", ".join(missing) if missing else str(directory)
        raise ModelCompatibilityError(f"Model '{model.id}' is incomplete: {detail}")
    return directory


__all__ = ["MODELS", "Model", "ModelCompatibilityError", "get_model", "model_options", "validate_model"]
