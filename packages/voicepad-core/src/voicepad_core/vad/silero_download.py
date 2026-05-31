# vad/silero_download.py

from __future__ import annotations

import urllib.request
from pathlib import Path

# ONNX model lives next to silero.py inside the package.
# This path is resolved relative to this file so it works
# regardless of where the process is launched from.
_VAD_DIR = Path(__file__).parent
_MODEL_FILENAME = "silero_vad.onnx"
MODEL_PATH = _VAD_DIR / _MODEL_FILENAME

# Direct ONNX file URL from the HuggingFace repo.
# onnx/silero_vad.onnx is the correct filename in that tree.
_DOWNLOAD_URL = "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/silero_vad.onnx"


def ensure_model_exists(verbose: bool = True) -> Path:
    """
    Check whether the Silero ONNX model file is present.
    If it is missing, download it from HuggingFace and save it
    next to this file at:

        vad/silero_vad.onnx

    This function is safe to call on every app startup — it is
    a no-op when the file already exists.

    Args:
        verbose: If True, print progress messages to stdout.
                 Set to False in tests or silent boot paths.

    Returns:
        Path to the model file (guaranteed to exist after this call).

    Raises:
        RuntimeError: If the download fails for any reason
                      (no internet, HuggingFace unreachable, etc.)
    """
    if MODEL_PATH.exists():
        if verbose:
            print(f"[VAD] Silero model found at: {MODEL_PATH}")
        return MODEL_PATH

    if verbose:
        print("[VAD] Silero ONNX model not found. Downloading...")
        print(f"      Source : {_DOWNLOAD_URL}")
        print(f"      Target : {MODEL_PATH}")

    _download(verbose=verbose)

    if verbose:
        size_kb = MODEL_PATH.stat().st_size // 1024
        print(f"[VAD] Download complete. ({size_kb} KB)")

    return MODEL_PATH


def _download(verbose: bool = True) -> None:
    """
    Stream the ONNX file from HuggingFace to disk with a
    simple progress indicator.

    Uses only stdlib urllib — no requests, no httpx dependency.

    Raises:
        RuntimeError: Wraps any urllib or IO error with a
                      clear human-readable message.
    """
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(_DOWNLOAD_URL) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192  # 8 KB per read

            with open(MODEL_PATH, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if verbose and total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r      Progress: {pct:3d}%", end="", flush=True)

        if verbose:
            print()  # newline after progress line

    except Exception as e:
        # Clean up a partial file if the download failed mid-way
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise RuntimeError(
            f"[VAD] Failed to download Silero ONNX model.\n"
            f"      URL   : {_DOWNLOAD_URL}\n"
            f"      Reason: {e}\n\n"
            "      Check your internet connection and try again."
        ) from e
