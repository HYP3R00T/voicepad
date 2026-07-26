from collections.abc import Callable
from pathlib import Path

from ..models import Model, ModelCompatibilityError, validate_model

ProgressCallback = Callable[[int, int | None], None]


class ArtifactError(RuntimeError):
    """Raised when a model cannot be downloaded or validated."""


def artifact_path(model: Model, cache_dir: Path) -> Path:
    """Return the stable local directory for a model."""
    return cache_dir.expanduser().resolve() / model.backend / model.id


def locate_artifact(model: Model, cache_dir: Path) -> Path | None:
    """Return a complete cached model without downloading it."""
    try:
        return validate_model(artifact_path(model, cache_dir), model)
    except ModelCompatibilityError:
        return None


def prepare_artifact(
    model: Model,
    cache_dir: Path,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download a pinned Hugging Face snapshot and validate its files."""
    target = artifact_path(model, cache_dir)
    cached = locate_artifact(model, cache_dir)
    if cached is not None:
        return cached

    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        snapshot_download(
            repo_id=model.repo,
            revision=model.revision,
            local_dir=str(target),
            allow_patterns=list(model.files),
        )
        result = validate_model(target, model)
    except Exception as exc:
        raise ArtifactError(f"Could not prepare model '{model.id}': {exc}") from exc

    if on_progress is not None:
        size = sum(path.stat().st_size for path in result.rglob("*") if path.is_file())
        on_progress(size, size)
    return result


__all__ = ["ArtifactError", "ProgressCallback", "locate_artifact", "prepare_artifact"]
