"""Tests for voicepad.cli.record."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner
from voicepad.cli.record import (
    _format_markdown,
    _print_result,
    _wait_for_quit,
    record_app,
)
from voicepad_core import AudioRecorderError, AudioTooShortError, TranscriptionError
from voicepad_core.config import Config

runner = CliRunner()


# ---------------------------------------------------------------------------
# _wait_for_quit
# ---------------------------------------------------------------------------


class TestWaitForQuit:
    def test_sets_event_on_q_input(self) -> None:
        stop_event = threading.Event()
        with patch("sys.stdin.readline", return_value="q\n"):
            _wait_for_quit(stop_event)
        assert stop_event.is_set()

    def test_sets_event_on_eof(self) -> None:
        stop_event = threading.Event()
        with patch("sys.stdin.readline", return_value=""):
            _wait_for_quit(stop_event)
        assert stop_event.is_set()

    def test_sets_event_on_eoferror(self) -> None:
        stop_event = threading.Event()
        with patch("sys.stdin.readline", side_effect=EOFError):
            _wait_for_quit(stop_event)
        assert stop_event.is_set()

    def test_sets_event_on_oserror(self) -> None:
        stop_event = threading.Event()
        with patch("sys.stdin.readline", side_effect=OSError("stdin closed")):
            _wait_for_quit(stop_event)
        assert stop_event.is_set()

    def test_ignores_non_q_input(self) -> None:
        stop_event = threading.Event()
        inputs = ["hello\n", "Q\n"]  # Q uppercase should work (case insensitive)
        with patch("sys.stdin.readline", side_effect=inputs):
            _wait_for_quit(stop_event)
        assert stop_event.is_set()


# ---------------------------------------------------------------------------
# _print_result
# ---------------------------------------------------------------------------


class TestPrintResult:
    def test_prints_transcription_result(self, capsys) -> None:
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.device = "cuda"
        mock_result.compute_type = "float16"
        mock_result.language = "en"
        mock_result.language_probability = 0.95
        mock_result.duration_s = 2.5
        mock_result.latency_ms = 150
        mock_result.fallback_to_cpu = False
        _print_result(mock_result)
        captured = capsys.readouterr()
        assert "Hello world" in captured.out
        assert "cuda" in captured.out
        assert "en" in captured.out

    def test_prints_fallback_warning(self, capsys) -> None:
        mock_result = MagicMock()
        mock_result.text = "Test"
        mock_result.device = "cpu"
        mock_result.compute_type = "int8"
        mock_result.language = "en"
        mock_result.language_probability = 0.9
        mock_result.duration_s = 1.0
        mock_result.latency_ms = 500
        mock_result.fallback_to_cpu = True
        _print_result(mock_result)
        captured = capsys.readouterr()
        assert "CUDA not available" in captured.out

    def test_handles_no_speech_detected(self, capsys) -> None:
        mock_result = MagicMock()
        mock_result.text = None
        mock_result.device = "cuda"
        mock_result.compute_type = "float16"
        mock_result.language = "en"
        mock_result.language_probability = 0.5
        mock_result.duration_s = 1.0
        mock_result.latency_ms = 100
        mock_result.fallback_to_cpu = False
        _print_result(mock_result)
        captured = capsys.readouterr()
        assert "no speech detected" in captured.out


# ---------------------------------------------------------------------------
# _format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    def test_formats_basic_result(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.device = "cuda"
        mock_result.compute_type = "float16"
        mock_result.language = "en"
        mock_result.language_probability = 0.95
        mock_result.duration_s = 2.5
        mock_result.latency_ms = 150
        mock_result.fallback_to_cpu = False
        mock_result.segments = []
        md = _format_markdown(wav_path, mock_result, "tiny")
        assert "test.wav" in md
        assert "Hello world" in md
        assert "tiny" in md
        assert "cuda" in md

    def test_includes_fallback_info(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        mock_result = MagicMock()
        mock_result.text = "Test"
        mock_result.device = "cpu"
        mock_result.compute_type = "int8"
        mock_result.language = "en"
        mock_result.language_probability = 0.9
        mock_result.duration_s = 1.0
        mock_result.latency_ms = 500
        mock_result.fallback_to_cpu = True
        mock_result.segments = []
        md = _format_markdown(wav_path, mock_result)
        assert "fallback: cpu" in md

    def test_includes_segments(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        seg1 = MagicMock()
        seg1.start, seg1.end, seg1.text = 0.0, 1.5, "Hello"
        seg2 = MagicMock()
        seg2.start, seg2.end, seg2.text = 1.5, 3.0, "world"
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.device = "cuda"
        mock_result.compute_type = "float16"
        mock_result.language = "en"
        mock_result.language_probability = 0.95
        mock_result.duration_s = 3.0
        mock_result.latency_ms = 150
        mock_result.fallback_to_cpu = False
        mock_result.segments = [seg1, seg2]
        md = _format_markdown(wav_path, mock_result)
        assert "## Segments" in md
        assert "[0.00s - 1.50s]" in md
        assert "Hello" in md
        assert "[1.50s - 3.00s]" in md
        assert "world" in md

    def test_handles_no_speech(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        mock_result = MagicMock()
        mock_result.text = None
        mock_result.device = "cuda"
        mock_result.compute_type = "float16"
        mock_result.language = "en"
        mock_result.language_probability = 0.5
        mock_result.duration_s = 1.0
        mock_result.latency_ms = 100
        mock_result.fallback_to_cpu = False
        mock_result.segments = []
        md = _format_markdown(wav_path, mock_result)
        assert "*(no speech detected)*" in md


# ---------------------------------------------------------------------------
# record start
# ---------------------------------------------------------------------------


class TestStartRecording:
    def test_exits_on_model_download_failure(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = np.zeros(16000, dtype=np.float32)  # 1s of audio
        mock_recorder.make_wav_path.return_value = tmp_path / "test.wav"
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=False),
            patch(
                "voicepad.cli.record.ensure_model_downloaded",
                side_effect=TranscriptionError("Download failed"),
            ),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.time.sleep"),
        ):
            result = runner.invoke(record_app, ["start", "--no-transcribe", "--duration", "0.5"])
        # When --no-transcribe is used, model download is skipped
        assert result.exit_code == 0

    def test_exits_on_model_load_failure(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", side_effect=TranscriptionError("Load failed")),
        ):
            result = runner.invoke(record_app, ["start"])
        assert result.exit_code == 1

    def test_exits_on_recorder_start_failure(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_recorder = MagicMock()
        mock_recorder.start.side_effect = AudioRecorderError("Device not found")
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cuda", "float16", False)),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
        ):
            result = runner.invoke(record_app, ["start"])
        assert result.exit_code == 1

    def test_skips_transcription_with_no_transcribe_flag(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = np.zeros(16000, dtype=np.float32)  # 1s of audio
        mock_recorder.make_wav_path.return_value = tmp_path / "test.wav"
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.time.sleep"),
        ):
            result = runner.invoke(record_app, ["start", "--no-transcribe", "--duration", "0.5"])
        assert result.exit_code == 0
        # Should not call transcription functions

    def test_skips_save_with_no_save_flag(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        mock_result = MagicMock()
        mock_result.text = "Test"
        mock_result.device = "cuda"
        mock_result.compute_type = "float16"
        mock_result.language = "en"
        mock_result.language_probability = 0.9
        mock_result.duration_s = 1.0
        mock_result.latency_ms = 100
        mock_result.fallback_to_cpu = False
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cuda", "float16", False)),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.transcribe_buffer", return_value=mock_result),
            patch("voicepad.cli.record.time.sleep"),
        ):
            result = runner.invoke(record_app, ["start", "--no-save", "--duration", "0.5"])
        assert result.exit_code == 0
        mock_recorder.save_wav.assert_not_called()

    def test_handles_audio_too_short_error(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        mock_recorder.make_wav_path.return_value = tmp_path / "test.wav"
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cuda", "float16", False)),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.transcribe_file", side_effect=AudioTooShortError("Too short")),
            patch("voicepad.cli.record.time.sleep"),
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.5"])
        assert result.exit_code == 0
        assert "SKIP" in result.stdout

    def test_exits_on_transcription_error(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        mock_recorder.make_wav_path.return_value = tmp_path / "test.wav"
        (tmp_path / "test.wav").write_bytes(b"fake wav")
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cuda", "float16", False)),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.transcribe_file", side_effect=TranscriptionError("Failed")),
            patch("voicepad.cli.record.time.sleep"),
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.5"])
        assert result.exit_code == 1

    def test_shows_fallback_warning_on_model_load(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", True)),
            patch("voicepad.cli.record.AudioRecorder") as mock_recorder_cls,
        ):
            mock_recorder = MagicMock()
            mock_recorder.start.side_effect = AudioRecorderError("Stop early")
            mock_recorder_cls.return_value = mock_recorder
            result = runner.invoke(record_app, ["start"])
        assert "CUDA not available" in result.stdout


# ---------------------------------------------------------------------------
# record info
# ---------------------------------------------------------------------------


class TestShowInfo:
    def test_displays_recording_info(self, tmp_path: Path) -> None:
        mock_config = Config(
            recordings_path=tmp_path,
            markdown_path=tmp_path,
            input_device_index=1,
            recording_prefix="test_",
        )
        with patch("voicepad.cli.record.get_config", return_value=mock_config):
            result = runner.invoke(record_app, ["info"])
        assert result.exit_code == 0
        assert "Recording" in result.stdout
        assert "Transcription" in result.stdout
        assert "input device" in result.stdout

    def test_shows_system_default_device(self, tmp_path: Path) -> None:
        mock_config = Config(recordings_path=tmp_path, markdown_path=tmp_path, input_device_index=None)
        with patch("voicepad.cli.record.get_config", return_value=mock_config):
            result = runner.invoke(record_app, ["info"])
        assert result.exit_code == 0
        assert "system default" in result.stdout
