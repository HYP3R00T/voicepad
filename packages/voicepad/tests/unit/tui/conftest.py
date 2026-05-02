"""Conftest for TUI unit tests.

Prevents background threads (pynput hotkey listener, recording timer) from
being spawned during tests. These threads survive past test teardown and cause
a Windows fatal exception (0x80000003) when the GC runs while pynput's win32
message loop is still active.

All TUI tests that exercise VoicePadApp get these patches automatically via
the ``no_background_threads`` autouse fixture.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_background_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress all background thread creation in VoicePadApp for every test.

    Patches:
    - ``_start_hotkey_listener`` — prevents pynput GlobalHotKeys thread
    - ``_start_timer`` / ``_stop_timer`` — prevents the recording timer thread
    - ``_check_first_run`` — prevents the setup modal from calling _warm_model_worker
      (which would start a worker thread that also triggers hotkey listener start)
    """
    from voicepad.tui.app import VoicePadApp

    monkeypatch.setattr(VoicePadApp, "_start_hotkey_listener", lambda self: None, raising=False)
    monkeypatch.setattr(VoicePadApp, "_start_timer", lambda self: None, raising=False)
    monkeypatch.setattr(VoicePadApp, "_stop_timer", lambda self: None, raising=False)
    monkeypatch.setattr(VoicePadApp, "_check_first_run", lambda self: None, raising=False)
