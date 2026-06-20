from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from voicepad_core.config import Config
from voicepad_core.logging_utils import (
    _cleanup_handlers,
    begin_transcription_session,
    configure_global_logging,
    end_transcription_session,
)


def _make_config(tmp_path: Path) -> Config:
    return Config(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
        logs_path=tmp_path / "logs",
    )


def test_configure_global_logging_uses_config_logs_path_when_not_provided(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    with patch("voicepad_core.logging_utils.get_config", return_value=config):
        log_file = configure_global_logging("INFO")
        logging.getLogger("voicepad.tests").info("hello from app logger")

    try:
        assert log_file.parent == config.logs_path
        assert log_file.name.startswith("voicepad-")
        assert log_file.suffix == ".log"
        assert log_file.exists()
        assert "hello from app logger" in log_file.read_text(encoding="utf-8")
    finally:
        _cleanup_handlers()


@patch("voicepad_core.streaming.transcriber.set_streaming_session_logger")
@patch("voicepad_core.inference.engine.set_session_logger")
def test_begin_transcription_session_wires_core_loggers(
    mock_set_session_logger, mock_set_streaming_logger, tmp_path: Path
) -> None:
    session_logger, log_file = begin_transcription_session(
        logs_path=tmp_path / "logs",
        log_level="DEBUG",
        session_id="abc123",
        include_streaming=True,
    )
    session_logger.info("scoped log works")

    try:
        mock_set_session_logger.assert_called_once_with(session_logger)
        mock_set_streaming_logger.assert_called_once_with(session_logger)
        assert log_file.parent == tmp_path / "logs" / "scopes" / "transcription"
        assert log_file.name.startswith("abc123-")
        assert log_file.suffix == ".log"
        assert log_file.exists()
        assert "scoped log works" in log_file.read_text(encoding="utf-8")
    finally:
        _cleanup_handlers()


@patch("voicepad_core.streaming.transcriber.set_streaming_session_logger")
@patch("voicepad_core.inference.engine.set_session_logger")
def test_end_transcription_session_clears_core_loggers(mock_set_session_logger, mock_set_streaming_logger) -> None:
    end_transcription_session(include_streaming=True)

    mock_set_session_logger.assert_called_once_with(None)
    mock_set_streaming_logger.assert_called_once_with(None)
