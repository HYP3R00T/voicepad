from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from utilityhub_logging import LogFormat, begin_scope_logging, bind_context, end_scope_logging, resolve_logs_path

APP_NAME = "voicepad"


@dataclass(slots=True)
class RecordingLogScope:
    """Own one recording log while allowing its records into the session log."""

    recording_id: str
    path: Path
    _logger: logging.Logger
    _closed: bool = False

    @classmethod
    def start(cls, recording_id: str) -> RecordingLogScope:
        if not recording_id.strip():
            raise ValueError("recording_id must be a non-empty string")

        logs_path = resolve_logs_path(app_name=APP_NAME, create=False)
        logs_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(logs_path, 0o700)

        context = bind_context(recording_id=recording_id)
        try:
            scope_logger, path = begin_scope_logging(
                "recording",
                recording_id,
                app_name=APP_NAME,
                logs_path=logs_path,
                log_format=LogFormat.JSON,
                propagate=True,
            )
        finally:
            context.close()

        try:
            for directory in (path.parent.parent, path.parent):
                os.chmod(directory, 0o700)
            os.chmod(path, 0o600)
        except Exception:
            end_scope_logging(scope_logger)
            raise

        scope = cls(recording_id, path, scope_logger)
        scope.logger_for("voicepad").info("Recording scope started")
        return scope

    @property
    def context(self) -> dict[str, str]:
        return {
            "recording_id": self.recording_id,
            "scope_id": self.recording_id,
            "scope_type": "recording",
        }

    def logger_for(self, component: str) -> logging.LoggerAdapter[logging.Logger]:
        context = {**self.context, "component": component}
        return logging.LoggerAdapter(
            self._logger.getChild(component),
            {"utilityhub_context": context},
        )

    def close(self, *, outcome: str, error: BaseException | None = None) -> None:
        if self._closed:
            return
        scope_logger = self.logger_for("voicepad")
        try:
            if error is None:
                scope_logger.info("Recording scope ended: outcome=%s", outcome)
            else:
                scope_logger.error(
                    "Recording scope ended: outcome=%s error_type=%s error=%s",
                    outcome,
                    type(error).__name__,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )
        finally:
            end_scope_logging(self._logger)
            self._closed = True


__all__ = ["APP_NAME", "RecordingLogScope"]
