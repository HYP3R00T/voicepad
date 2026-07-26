"""Configure packaged NVIDIA DLLs immediately before CUDA runtime use."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_configuration_lock = Lock()
_configured = False
_dll_directory_handles: list[Any] = []


def configure_windows_cuda_dlls() -> None:
    """Expose packaged NVIDIA DLLs to native runtimes once on Windows.

    Importing :mod:`voicepad_core` remains side-effect free. Native DLL
    discovery is deferred until a CUDA-capable backend is opened.
    """
    global _configured

    if sys.platform != "win32" or _configured:
        return

    with _configuration_lock:
        if _configured:
            return

        dll_directories = _find_nvidia_dll_directories()
        for directory in dll_directories:
            _dll_directory_handles.append(os.add_dll_directory(str(directory)))
            _prepend_to_path(directory)

        remaining = [dll for directory in dll_directories for dll in sorted(directory.glob("*.dll"))]
        loaded_count = 0
        for _ in range(2):
            still_remaining: list[Path] = []
            for dll in remaining:
                try:
                    ctypes.WinDLL(str(dll))
                    loaded_count += 1
                except OSError:
                    still_remaining.append(dll)
            remaining = still_remaining
            if not remaining:
                break

        _configured = True
        logger.debug(
            "Configured %d NVIDIA DLL directories and preloaded %d DLLs",
            len(dll_directories),
            loaded_count,
        )


def _find_nvidia_dll_directories() -> list[Path]:
    """Return unique packaged NVIDIA DLL directories from the import path."""
    for path_entry in sys.path:
        nvidia_directory = Path(path_entry) / "nvidia"
        if nvidia_directory.is_dir():
            return sorted({dll.parent for dll in nvidia_directory.rglob("*.dll")})
    return []


def _prepend_to_path(directory: Path) -> None:
    """Prepend a DLL directory to PATH unless it is already present."""
    directory_string = str(directory)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if directory_string in entries:
        return
    os.environ["PATH"] = os.pathsep.join([directory_string, *entries])
