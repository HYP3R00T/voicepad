from __future__ import annotations

import logging
import sys


def main() -> int | None:
    """Dispatch the lightweight toggle command or launch the full CLI."""
    from voicepad.diagnostics import configure_logging

    configure_logging()
    logger = logging.getLogger(__name__)
    try:
        if sys.argv[1:] == ["toggle"]:
            from voicepad.tui.control import run_toggle_command

            return run_toggle_command()

        from voicepad.main import app

        app()
        return None
    finally:
        logger.info("Application session stopped")
