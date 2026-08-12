from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from pytest import MonkeyPatch
from voicepad.diagnostics import configure_logging


def test_configure_logging_creates_private_durable_session_log(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_exception_hook = sys.excepthook
    original_thread_hook = threading.excepthook
    try:
        destination = configure_logging()
        logging.getLogger("voicepad.test").error(
            "capture failed: elapsed_s=30.0 persisted_frames=16000 persisted_duration_s=1.0"
        )
        for handler in root.handlers:
            handler.flush()

        assert destination.parent == tmp_path / "voicepad" / "logs"
        assert destination.stat().st_mode & 0o077 == 0
        content = destination.read_text(encoding="utf-8")
        assert "Application session started" in content
        assert "elapsed_s=30.0 persisted_frames=16000 persisted_duration_s=1.0" in content
        assert "transcript" not in content
    finally:
        for handler in root.handlers:
            if handler not in original_handlers:
                handler.close()
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        sys.excepthook = original_exception_hook
        threading.excepthook = original_thread_hook
