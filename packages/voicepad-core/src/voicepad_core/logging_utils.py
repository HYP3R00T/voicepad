"""Thin logging adapter for VoicePad built on top of utilityhub-logging."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from utilityhub_logging import LogFormat, begin_scope_logging, cleanup_logging, configure_app_logging, end_scope_logging

from .config import get_config

APP_NAME = "voicepad"
_active_scope_logger: logging.Logger | None = None
_extra_scope_handlers: dict[str, list[logging.Handler]] = {}


def _resolve_logs_path(logs_path: Path | None = None) -> Path:
    """Resolve and create the configured logs directory."""
    resolved = logs_path if logs_path is not None else get_config().logs_path
    resolved = Path(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _generate_session_id() -> str:
    """Generate a high-resolution session identifier for log file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def setup_transcription_logger(
    logs_path: Path | None = None,
    log_level: str = "INFO",
    session_id: str | None = None,
    structured: bool = False,
    use_queue: bool = False,
    console: bool = False,
) -> tuple[logging.Logger, Path]:
    """Create one scoped transcription logger via utilityhub-logging."""
    if session_id is None:
        session_id = _generate_session_id()

    logger, log_file = begin_scope_logging(
        "transcription",
        session_id,
        app_name=APP_NAME,
        level=log_level,
        logs_path=_resolve_logs_path(logs_path),
        log_format=LogFormat.JSON if structured else LogFormat.PLAIN,
        logger=logging.getLogger(f"{APP_NAME}.transcription.{session_id}"),
        propagate=False,
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
        logger.addHandler(console_handler)
        _extra_scope_handlers.setdefault(logger.name, []).append(console_handler)

    if use_queue:
        logger.warning(
            "Queue-backed transcription logging is not supported by utilityhub-logging; using direct handlers."
        )

    logger.info("Transcription session started: %s", session_id)
    logger.info("Log file: %s", log_file)
    return logger, log_file


def begin_transcription_session(
    logs_path: Path | None = None,
    log_level: str = "INFO",
    session_id: str | None = None,
    *,
    include_streaming: bool = False,
    structured: bool = False,
    use_queue: bool = False,
    console: bool = False,
) -> tuple[logging.Logger, Path]:
    """Create a scoped logger and attach it to the transcription subsystems."""
    global _active_scope_logger

    session_logger, log_file = setup_transcription_logger(
        logs_path=logs_path,
        log_level=log_level,
        session_id=session_id,
        structured=structured,
        use_queue=use_queue,
        console=console,
    )
    _active_scope_logger = session_logger

    from .inference.engine import set_session_logger

    set_session_logger(session_logger)

    if include_streaming:
        from .streaming.transcriber import set_streaming_session_logger

        set_streaming_session_logger(session_logger)

    return session_logger, log_file


def end_transcription_session(*, include_streaming: bool = False) -> None:
    """Detach and close the current scoped transcription logger."""
    global _active_scope_logger

    from .inference.engine import set_session_logger

    set_session_logger(None)

    if include_streaming:
        from .streaming.transcriber import set_streaming_session_logger

        set_streaming_session_logger(None)

    if _active_scope_logger is None:
        return

    for handler in _extra_scope_handlers.pop(_active_scope_logger.name, []):
        _active_scope_logger.removeHandler(handler)
        handler.close()
    end_scope_logging(_active_scope_logger)
    _active_scope_logger = None


def _cleanup_handlers() -> None:
    """Flush and close managed logging handlers."""
    for logger_name, handlers in list(_extra_scope_handlers.items()):
        logger = logging.getLogger(logger_name)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        _extra_scope_handlers.pop(logger_name, None)
    cleanup_logging(close_all_loggers=True)


def configure_global_logging(
    log_level: str = "INFO",
    logs_path: Path | None = None,
    console: bool = False,
    session_id: str | None = None,
    session_prefix: str = "session",
) -> Path:
    """Configure one app-session logger for the current VoicePad process."""
    del session_id
    del session_prefix

    log_file = configure_app_logging(
        APP_NAME,
        level=log_level,
        logs_path=_resolve_logs_path(logs_path),
        console=console,
        log_format=LogFormat.PLAIN,
        logger=logging.getLogger(),
        propagate=False,
    )
    logging.getLogger(__name__).info("Application session log: %s", log_file)
    return log_file


def log_transcription_start(
    logger: logging.Logger,
    audio_duration: float,
    model_name: str,
    device: str,
    compute_type: str,
) -> None:
    """Log the start of a transcription operation."""
    logger.info(
        "Transcription started: duration=%.2fs model=%s device=%s compute=%s",
        audio_duration,
        model_name,
        device,
        compute_type,
    )


def log_transcription_end(
    logger: logging.Logger,
    success: bool,
    latency_ms: float | None = None,
    text_length: int | None = None,
    error: str | None = None,
) -> None:
    """Log the end of a transcription operation."""
    if success:
        logger.info(
            "Transcription completed: latency_ms=%s text_length=%s",
            f"{latency_ms:.0f}" if latency_ms is not None else "n/a",
            text_length if text_length is not None else "n/a",
        )
        return

    logger.error("Transcription failed: %s", error or "unknown error")
