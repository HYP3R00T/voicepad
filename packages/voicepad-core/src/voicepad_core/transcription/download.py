"""Model download and cache management."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path as _Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from .constants import HF_REPO_PREFIX
from .exceptions import TranscriptionError

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)


def _get_repo_id(model_name: str) -> str:
    """Resolve HuggingFace repository ID for model name.

    Uses faster-whisper's internal model registry for accurate resolution.
    Falls back to Systran prefix for unknown models.

    Args:
        model_name: Whisper model name

    Returns:
        HuggingFace repository ID
    """
    try:
        from faster_whisper.utils import _MODELS  # type: ignore[attr-defined]

        if model_name in _MODELS:
            return _MODELS[model_name]
    except Exception:
        pass
    return f"{HF_REPO_PREFIX}{model_name}"


def model_downloaded(model_name: str, config: Config | None = None) -> bool:
    """Check if model weights exist in local cache.

    Verifies presence of actual weight file, not just metadata.
    No network requests are made.

    Args:
        model_name: Whisper model name
        config: Optional config for custom cache path

    Returns:
        True if model weights are cached locally
    """
    if config is not None:
        cache_root = config.model_cache_path / "hub"
    else:
        hf_home = os.environ.get("HF_HOME", "")
        cache_root = _Path(hf_home) / "hub" if hf_home else _Path.home() / ".cache" / "huggingface" / "hub"

    repo_id = _get_repo_id(model_name)
    snapshots = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"

    if not snapshots.exists():
        return False

    return any(snap.is_dir() and (snap / "model.bin").exists() for snap in snapshots.iterdir())


def ensure_model_downloaded(
    model_name: str,
    config: Config | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Download model weights if not cached.

    Blocks until download completes. Uses HuggingFace Hub for downloads.

    Args:
        model_name: Whisper model name
        config: Optional config for custom cache path
        on_progress: Optional callback(downloaded_bytes, total_bytes) for progress tracking

    Raises:
        TranscriptionError: If download fails
    """
    if model_downloaded(model_name, config):
        return

    repo_id = _get_repo_id(model_name)
    logger.info(f"Downloading '{model_name}' from {repo_id}")

    cache_dir: str | None = None
    if config is not None:
        hub_dir = config.model_cache_path / "hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(hub_dir)

    tqdm_class = None
    if on_progress is not None:
        _cb = on_progress
        from tqdm.auto import tqdm as _tqdm

        _downloaded = [0]
        _total = [0]

        class _ProgressTqdm(_tqdm):  # ty:ignore[unsupported-base]
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

        tqdm_class = _ProgressTqdm

    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
            tqdm_class=tqdm_class,
        )
    except HfHubHTTPError as e:
        raise TranscriptionError(f"Failed to download model '{model_name}': {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Failed to download model '{model_name}': {e}") from e
