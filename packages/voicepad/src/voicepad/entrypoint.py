from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from utilityhub_logging import LogFormat, cleanup_logging, configure_app_logging, resolve_logs_path

from .observability import APP_NAME

logger = logging.getLogger(__name__)


def _configure_logging() -> Path:
    log_dir = resolve_logs_path(app_name=APP_NAME, create=False)
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(log_dir, 0o700)
    log_path = configure_app_logging(
        app_name=APP_NAME,
        level="INFO",
        logs_path=log_dir,
        console=False,
        log_format=LogFormat.JSON,
    )
    os.chmod(log_path, 0o600)
    logger.info("Application session started: pid=%s log=%s", os.getpid(), log_path)
    return log_path


def main() -> int | None:
    """Dispatch the lightweight toggle command or launch the full CLI."""
    _configure_logging()
    try:
        if sys.argv[1:] == ["toggle"]:
            from voicepad.tui.control import run_toggle_command

            return run_toggle_command()

        from voicepad.main import app

        app()
        return None
    finally:
        error = sys.exception()
        if error is None:
            logger.info("Application session ended: outcome=completed")
        else:
            logger.error(
                "Application session ended: outcome=failed error_type=%s error=%s",
                type(error).__name__,
                error,
            )
        cleanup_logging()
