from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_configuration_lock = Lock()
_configured = False
_library_handles: list[Any] = []


def configure_linux_cuda_libraries() -> None:
    """Preload packaged NVIDIA shared libraries once on Linux."""
    global _configured

    if sys.platform != "linux" or _configured:
        return

    with _configuration_lock:
        if _configured:
            return

        remaining = [
            library for directory in _find_nvidia_library_directories() for library in sorted(directory.glob("*.so*"))
        ]
        for _ in range(2):
            still_remaining: list[Path] = []
            for library in remaining:
                try:
                    _library_handles.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL))
                except OSError:
                    still_remaining.append(library)
            remaining = still_remaining
            if not remaining:
                break

        _configured = True
        logger.debug(
            "Preloaded %d packaged NVIDIA shared libraries; %d could not be loaded",
            len(_library_handles),
            len(remaining),
        )


def _find_nvidia_library_directories() -> list[Path]:
    for path_entry in sys.path:
        nvidia_directory = Path(path_entry) / "nvidia"
        if nvidia_directory.is_dir():
            return sorted({library.parent for library in nvidia_directory.rglob("*.so*")})
    return []
