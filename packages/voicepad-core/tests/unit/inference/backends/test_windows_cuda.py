"""Tests for deferred Windows CUDA DLL configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from voicepad_core.inference.backends import windows_cuda


@pytest.fixture(autouse=True)
def reset_configuration_state():
    """Reset process-wide helper state around each isolated unit test."""
    windows_cuda._configured = False
    windows_cuda._dll_directory_handles.clear()
    yield
    windows_cuda._configured = False
    windows_cuda._dll_directory_handles.clear()


def test_configure_noops_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-Windows hosts do not scan for or configure native DLLs."""
    finder = Mock()
    monkeypatch.setattr(windows_cuda.sys, "platform", "linux")
    monkeypatch.setattr(windows_cuda, "_find_nvidia_dll_directories", finder)

    windows_cuda.configure_windows_cuda_dlls()

    finder.assert_not_called()


def test_configure_is_idempotent_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated Windows setup calls register each DLL directory only once."""
    directory = Path("C:/python/site-packages/nvidia/cublas/bin")
    add_directory = Mock(return_value=object())
    monkeypatch.setattr(windows_cuda.sys, "platform", "win32")
    monkeypatch.setattr(
        windows_cuda,
        "_find_nvidia_dll_directories",
        lambda: [directory],
    )
    monkeypatch.setattr(windows_cuda.os, "add_dll_directory", add_directory)

    windows_cuda.configure_windows_cuda_dlls()
    windows_cuda.configure_windows_cuda_dlls()

    add_directory.assert_called_once_with(str(directory))
