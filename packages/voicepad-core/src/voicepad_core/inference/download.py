# inference/download.py

"""Model download and local cache management.

Models are downloaded from HuggingFace via huggingface_hub and stored under
the configured VoicePad model cache directory.

The public surface is two functions:
    model_downloaded(model_name)          -> bool
    ensure_model_downloaded(model_name)   -> Path
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from .errors import ModelNotFoundError
from ..config import get_config
from ..models import ModelCompatibilityError, resolve_model_spec, validate_model_snapshot

logger = logging.getLogger(__name__)


def _get_models_dir(models_dir: Path | None = None) -> Path:
    """Return the configured model cache directory or an explicit override."""
    if models_dir is not None:
        return models_dir
    return get_config().model_cache_path


def _get_cache_roots(models_dir: Path | None = None) -> tuple[Path, ...]:
    """Return cache roots to inspect, newest HuggingFace layout first."""
    base_dir = _get_models_dir(models_dir)
    return (base_dir / "hub", base_dir)


def _get_repo_id(model_name: str) -> str:
    """Resolve the HuggingFace repository ID for a given model name."""
    return resolve_model_spec(model_name).repo_id


def _find_model_snapshot(model_name: str, models_dir: Path | None = None) -> Path | None:
    """Return the most recent cached model snapshot path, if present."""
    repo_id = _get_repo_id(model_name)
    repo_cache_dir = f"models--{repo_id.replace('/', '--')}"
    snapshots_found: list[Path] = []

    for cache_root in _get_cache_roots(models_dir):
        snapshots = cache_root / repo_cache_dir / "snapshots"
        if not snapshots.exists():
            continue
        snapshots_found.extend(snap for snap in snapshots.iterdir() if snap.is_dir() and (snap / "model.bin").exists())

    if not snapshots_found:
        return None

    return max(snapshots_found, key=lambda snap: snap.stat().st_mtime)


def model_downloaded(model_name: str, models_dir: Path | None = None) -> bool:
    """Check whether compatible model weights exist in the local cache."""
    snapshot = _find_model_snapshot(model_name, models_dir)
    if snapshot is None:
        return False

    try:
        validate_model_snapshot(snapshot)
    except ModelCompatibilityError:
        return False

    return True


def ensure_model_downloaded(
    model_name: str,
    models_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download model weights from HuggingFace if not already cached."""
    cache_dir = _get_models_dir(models_dir)

    cached_snapshot = _find_model_snapshot(model_name, cache_dir)
    if cached_snapshot is not None:
        try:
            validate_model_snapshot(cached_snapshot)
        except ModelCompatibilityError as e:
            raise ModelNotFoundError(f"Cached model '{model_name}' is not compatible: {e}") from e
        logger.debug(f"Model '{model_name}' already cached — skipping download.")
        return cached_snapshot

    model_spec = resolve_model_spec(model_name)
    repo_id = model_spec.repo_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading '{model_name}' from {repo_id} → {cache_dir}")

    tqdm_class = None
    if on_progress is not None:
        tqdm_class = _make_progress_tqdm(on_progress)

    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            ignore_patterns=[
                "*.msgpack",
                "*.h5",
                "flax_model*",
                "tf_model*",
                "rust_model*",
            ],
            tqdm_class=tqdm_class,
            revision=model_spec.revision,
        )
        logger.info(f"Model '{model_name}' downloaded successfully.")

        try:
            repo_cache_dir = f"models--{repo_id.replace('/', '--')}"
            snapshots_dir = cache_dir / repo_cache_dir / "snapshots"
            if snapshots_dir.exists():
                snaps = [d for d in snapshots_dir.iterdir() if d.is_dir() and (d / "model.bin").exists()]
                if snaps:
                    used = max(snaps, key=lambda p: p.stat().st_mtime)
                    total_bytes = sum(f.stat().st_size for f in used.rglob("*") if f.is_file())
                    logger.info(f"Snapshot used: {used} ({total_bytes} bytes)")
        except Exception:
            logger.debug("Could not determine snapshot directory after download", exc_info=True)

        snapshot = _find_model_snapshot(model_name, cache_dir)
        if snapshot is None:
            raise ModelNotFoundError(f"Failed to locate cached snapshot for model '{model_name}' after download.")
        validate_model_snapshot(snapshot)
        return snapshot

    except HfHubHTTPError as e:
        raise ModelNotFoundError(
            f"Failed to download model '{model_name}' from HuggingFace.\n"
            f"  repo  : {repo_id}\n"
            f"  reason: {e}\n\n"
            "Check your internet connection and try again."
        ) from e
    except ModelCompatibilityError as e:
        raise ModelNotFoundError(f"Downloaded model '{model_name}' is not compatible: {e}") from e
    except Exception as e:
        raise ModelNotFoundError(f"Failed to download model '{model_name}'.\n  reason: {e}") from e


def _make_progress_tqdm(callback: Callable[[int, int], None]):
    """Build a tqdm subclass that fires callback(downloaded, total)."""
    from tqdm.auto import tqdm as _tqdm

    _downloaded = [0]
    _total = [0]
    _cb = callback

    class _ProgressTqdm(_tqdm):  # type: ignore
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs.setdefault("disable", True)
            super().__init__(*args, **kwargs)
            self._track = bool(self.total and int(self.total) > 1_000_000)
            if self._track and _total[0] == 0 and self.total:
                _total[0] = int(self.total)

        def update(self, n: int = 1) -> bool | None:
            result = super().update(n)
            if self._track and n and n > 0:
                _downloaded[0] += n
                _cb(_downloaded[0], _total[0])
            return result

    return _ProgressTqdm
