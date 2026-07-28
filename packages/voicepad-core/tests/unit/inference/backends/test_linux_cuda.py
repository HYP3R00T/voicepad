from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from voicepad_core.inference.backends import linux_cuda


@pytest.fixture(autouse=True)
def reset_configuration_state():
    """Reset process-wide helper state around each isolated unit test."""
    linux_cuda._configured = False
    linux_cuda._library_handles.clear()
    yield
    linux_cuda._configured = False
    linux_cuda._library_handles.clear()


def test_configure_noops_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-Linux hosts do not scan for or preload native shared libraries."""
    finder = Mock()
    monkeypatch.setattr(linux_cuda.sys, "platform", "win32")
    monkeypatch.setattr(linux_cuda, "_find_nvidia_library_directories", finder)

    linux_cuda.configure_linux_cuda_libraries()

    finder.assert_not_called()


def test_configure_retries_dependencies_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux setup retries dependent libraries and configures the process once."""
    directory = Path("/python/site-packages/nvidia/cublas/lib")
    dependent = directory / "liba-dependent.so.12"
    dependency = directory / "libz-dependency.so.12"
    load_attempts: list[Path] = []

    def load_library(path: str, *, mode: int) -> object:
        library = Path(path)
        load_attempts.append(library)
        if library == dependent and load_attempts.count(dependent) == 1:
            raise OSError("dependency not loaded")
        assert mode == linux_cuda.ctypes.RTLD_GLOBAL
        return object()

    monkeypatch.setattr(linux_cuda.sys, "platform", "linux")
    monkeypatch.setattr(linux_cuda, "_find_nvidia_library_directories", lambda: [directory])
    monkeypatch.setattr(Path, "glob", lambda _self, _pattern: iter((dependent, dependency)))
    monkeypatch.setattr(linux_cuda.ctypes, "CDLL", load_library)

    linux_cuda.configure_linux_cuda_libraries()
    linux_cuda.configure_linux_cuda_libraries()

    assert load_attempts == [dependent, dependency, dependent]
