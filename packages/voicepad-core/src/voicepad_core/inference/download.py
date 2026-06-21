# inference/download.py

"""Model download and local cache management.

Models are downloaded from HuggingFace via huggingface_hub and stored under
the configured VoicePad model cache directory.

The public surface is two functions:
    model_downloaded(model_name)          -> bool
    ensure_model_downloaded(model_name)   -> None
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from .constants import HF_REPO_PREFIX
from .errors import ModelNotFoundError
from ..config import get_config

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


# ---------------------------------------------------------------------------
# Repo resolution
# ---------------------------------------------------------------------------


def _get_repo_id(model_name: str) -> str:
    """Resolve the HuggingFace repository ID for a given model name.

    Uses faster-whisper's internal registry first for accurate resolution.
    Falls back to the Systran prefix for unknown or custom model names.

    Args:
        model_name: Whisper model name (e.g. 'turbo', 'large-v3').

    Returns:
        HuggingFace repository ID string.
    """
    try:
        from faster_whisper.utils import _MODELS  # type: ignore[attr-defined]

        if model_name in _MODELS:
            return _MODELS[model_name]
    except Exception:
        pass

    return f"{HF_REPO_PREFIX}{model_name}"


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------


def model_downloaded(model_name: str, models_dir: Path | None = None) -> bool:
    """Check whether model weights exist in the local cache.

    Verifies presence of the actual weight file (model.bin), not just
    metadata or snapshot directories. Makes no network requests.

    Args:
        model_name: Whisper model name (e.g. 'turbo').
        models_dir: Override the default models directory.
                Defaults to the configured model_cache_path.

    Returns:
        True if the model weights are fully cached locally.
    """
    repo_id = _get_repo_id(model_name)
    repo_cache_dir = f"models--{repo_id.replace('/', '--')}"

    for cache_root in _get_cache_roots(models_dir):
        snapshots = cache_root / repo_cache_dir / "snapshots"

        if not snapshots.exists():
            continue

        if any(snap.is_dir() and (snap / "model.bin").exists() for snap in snapshots.iterdir()):
            return True

    return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def ensure_model_downloaded(
    model_name: str,
    models_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Download model weights from HuggingFace if not already cached.

    Blocks until the download is complete. Safe to call on every startup —
    it is a no-op when the model is already present.

    Downloaded files are stored at:
        <model_cache_path>/models--<repo_id>/snapshots/<hash>/

    Ignored file patterns (not downloaded):
        *.msgpack, *.h5, flax_model*, tf_model*, rust_model*

    Args:
        model_name:   Whisper model name (e.g. 'turbo').
        models_dir:   Override the default models directory.
        on_progress:  Optional callback(downloaded_bytes, total_bytes).
                      Called during download for progress tracking.

    Raises:
        ModelNotFoundError: If the download fails for any reason
                            (network error, HuggingFace unavailable, etc.)
    """
    cache_dir = _get_models_dir(models_dir)

    if model_downloaded(model_name, cache_dir):
        logger.debug(f"Model '{model_name}' already cached — skipping download.")
        return

    repo_id = _get_repo_id(model_name)
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
        )
        logger.info(f"Model '{model_name}' downloaded successfully.")

        # Attempt to detect which snapshot directory was used and log it
        try:
            repo_cache_dir = f"models--{repo_id.replace('/', '--')}"
            snapshots_dir = cache_dir / repo_cache_dir / "snapshots"
            if snapshots_dir.exists():
                snaps = [d for d in snapshots_dir.iterdir() if d.is_dir() and (d / "model.bin").exists()]
                if snaps:
                    # Pick the most recent snapshot by mtime
                    used = max(snaps, key=lambda p: p.stat().st_mtime)
                    total_bytes = sum(f.stat().st_size for f in used.rglob("*") if f.is_file())
                    logger.info(f"Snapshot used: {used} ({total_bytes} bytes)")
        except Exception:
            logger.debug("Could not determine snapshot directory after download", exc_info=True)

    except HfHubHTTPError as e:
        raise ModelNotFoundError(
            f"Failed to download model '{model_name}' from HuggingFace.\n"
            f"  repo  : {repo_id}\n"
            f"  reason: {e}\n\n"
            "Check your internet connection and try again."
        ) from e

    except Exception as e:
        raise ModelNotFoundError(f"Failed to download model '{model_name}'.\n  reason: {e}") from e


# ---------------------------------------------------------------------------
# Internal — progress tqdm wrapper
# ---------------------------------------------------------------------------


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
