# inference/download.py

"""Model download and local cache management.

Models are downloaded from HuggingFace via huggingface_hub and stored at:

    inference/models/<model_name>/

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
from .exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)

# Models are stored here, co-located with this package.
_MODELS_DIR = Path(__file__).parent / "models"


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
                    Defaults to inference/models/.

    Returns:
        True if the model weights are fully cached locally.
    """
    cache_root = models_dir or _MODELS_DIR
    repo_id = _get_repo_id(model_name)
    snapshots = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"

    if not snapshots.exists():
        return False

    return any(snap.is_dir() and (snap / "model.bin").exists() for snap in snapshots.iterdir())


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
        inference/models/models--<repo_id>/snapshots/<hash>/

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
    if model_downloaded(model_name, models_dir):
        logger.debug(f"Model '{model_name}' already cached — skipping download.")
        return

    repo_id = _get_repo_id(model_name)
    cache_dir = models_dir or _MODELS_DIR
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
