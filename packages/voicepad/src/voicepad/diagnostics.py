from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from uuid import uuid4

_LOG_FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s - %(message)s"


def logs_path() -> Path:
    """Return the private directory containing durable application session logs."""
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "voicepad" / "logs"


def configure_logging() -> Path:
    """Create one durable log for this process before application work begins."""
    directory = logs_path()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = directory / f"voicepad-{stamp}-{uuid4().hex[:8]}.log"

    handler = logging.FileHandler(destination, encoding="utf-8")
    os.chmod(destination, 0o600)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    def log_uncaught(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        logging.getLogger("voicepad.crash").critical(
            "Uncaught application exception",
            exc_info=(exception_type, exception, traceback),
        )
        sys.__excepthook__(exception_type, exception, traceback)

    def log_thread_crash(args: threading.ExceptHookArgs) -> None:
        exc_info = None
        if args.exc_value is not None:
            exc_info = (args.exc_type, args.exc_value, args.exc_traceback)
        logging.getLogger("voicepad.crash").critical(
            "Uncaught thread exception: thread=%s",
            args.thread.name if args.thread is not None else "unknown",
            exc_info=exc_info,
        )

    sys.excepthook = log_uncaught
    threading.excepthook = log_thread_crash
    logging.getLogger(__name__).info("Application session started: pid=%s log=%s", os.getpid(), destination)
    return destination
