# vad/silero_download.py

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from ..config import Config, get_config

logger = logging.getLogger(__name__)


def get_model_path(vad_model_dir: Path | None = None, config: Config | None = None) -> Path:
    """
    Get the path where the VAD model should be stored.

    Args:
        vad_model_dir: Directory for VAD models. If None, uses config default.

    Returns:
        Full path to the VAD model file.
    """
    resolved_config = config or get_config()
    if vad_model_dir is None:
        vad_model_dir = resolved_config.vad_model_path

    return vad_model_dir / resolved_config.vad_model_filename


def ensure_model_exists(
    vad_model_dir: Path | None = None,
    verbose: bool = True,
    config: Config | None = None,
) -> Path:
    """
    Check whether the Silero ONNX model file is present.
    If it is missing, download it from the configured location.

    This function is safe to call on every app startup — it is
    a no-op when the file already exists.

    Args:
        vad_model_dir: Directory for VAD models. If None, uses config default.
        verbose: If True, print progress messages to stdout.
                 Set to False in tests or silent boot paths.

    Returns:
        Path to the model file (guaranteed to exist after this call).

    Raises:
        RuntimeError: If the download fails for any reason
                      (no internet, network unreachable, etc.)
    """
    resolved_config = config or get_config()
    model_path = get_model_path(vad_model_dir, config=resolved_config)

    if model_path.exists():
        if verbose:
            logger.info(f"[VAD] Silero model found at: {model_path}")
        return model_path

    if verbose:
        logger.info("[VAD] Silero ONNX model not found. Downloading...")
        logger.info(f"      Source : {resolved_config.vad_model_url}")
        logger.info(f"      Target : {model_path}")

    _download(
        model_path,
        download_url=resolved_config.vad_model_url,
        chunk_size=resolved_config.vad_download_chunk_size,
        verbose=verbose,
    )

    if verbose:
        size_kb = model_path.stat().st_size // 1024
        print(f"[VAD] Download complete. ({size_kb} KB)")

    return model_path


def _download(
    model_path: Path,
    download_url: str,
    chunk_size: int,
    verbose: bool = True,
) -> None:
    """
    Stream the ONNX file from HuggingFace to disk with a
    simple progress indicator.

    Uses only stdlib urllib — no requests, no httpx dependency.

    Raises:
        RuntimeError: Wraps any urllib or IO error with a
                      clear human-readable message.
    """
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(download_url) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(model_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if verbose and total > 0:
                        pct = downloaded * 100 // total
                        # Use debug-level progress updates to avoid spamming INFO logs
                        logger.debug(f"[VAD] Progress: {pct:3d}%")

        if verbose:
            logger.debug("[VAD] Download completed stream read")

    except Exception as e:
        # Clean up a partial file if the download failed mid-way
        if model_path.exists():
            try:
                model_path.unlink()
            except Exception:
                logger.exception("Failed to remove partial VAD model file")
        logger.exception("Failed to download Silero ONNX model")
        raise RuntimeError(
            f"[VAD] Failed to download Silero ONNX model.\n"
            f"      URL   : {download_url}\n"
            f"      Reason: {e}\n\n"
            "      Check your internet connection and try again."
        ) from e
