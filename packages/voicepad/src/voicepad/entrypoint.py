from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def _configure_logging() -> None:
    log_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicepad/logs"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path = log_dir / f"voicepad-{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}-{os.getpid()}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    os.chmod(log_path, 0o600)
    logging.info("Application session started: pid=%s log=%s", os.getpid(), log_path)


def main() -> int | None:
    """Dispatch the lightweight toggle command or launch the full CLI."""
    _configure_logging()
    if sys.argv[1:] == ["toggle"]:
        from voicepad.tui.control import run_toggle_command

        return run_toggle_command()

    from voicepad.main import app

    app()
    return None
