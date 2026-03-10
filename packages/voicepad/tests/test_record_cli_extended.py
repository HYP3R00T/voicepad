"""Additional unit tests for record CLI flows."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voicepad-core" / "src"))

from voicepad.cli.record import _transcribe_audio_file, _wait_for_quit_key, show_info, start_recording
from voicepad_core.recorder import AudioRecorderError


class RecordCliExtendedTests(unittest.TestCase):
    def test_wait_for_quit_key_sets_event(self) -> None:
        event = threading.Event()
        with patch("voicepad.cli.record.sys.stdin.readline", return_value="q\n"):
            _wait_for_quit_key(event)
        self.assertTrue(event.is_set())

    def test_wait_for_quit_key_handles_stdin_error(self) -> None:
        event = threading.Event()
        with patch("voicepad.cli.record.sys.stdin.readline", side_effect=EOFError):
            _wait_for_quit_key(event)
        self.assertTrue(event.is_set())

    def test_transcribe_audio_file_success_and_errors(self) -> None:
        stats = {
            "language": "en",
            "language_probability": 0.9,
            "duration": 2.0,
            "word_count": 3,
            "segment_count": 1,
            "device": "cpu",
            "fallback_info": {"fallback_occurred": False},
        }
        cfg = SimpleNamespace(markdown_path=Path("data/markdown"))

        with (
            patch("voicepad_core.transcribe_audio", return_value=stats),
            patch("pathlib.Path.mkdir"),
            patch("voicepad.cli.record.typer.secho") as mock_secho,
        ):
            _transcribe_audio_file(Path("x.wav"), cfg)
        self.assertTrue(mock_secho.called)

        with (
            patch("voicepad_core.transcribe_audio", side_effect=Exception("boom")),
            patch("pathlib.Path.mkdir"),
            patch("voicepad.cli.record.logger.exception") as mock_log,
        ):
            _transcribe_audio_file(Path("x.wav"), cfg)
        self.assertTrue(mock_log.called)

    def test_start_recording_fixed_duration_path(self) -> None:
        cfg = SimpleNamespace(
            input_device_index=1,
            recordings_path=Path("data/recordings"),
            recording_prefix="recording",
            vad_enabled=False,
            transcription_model="tiny",
        )
        recorder = MagicMock()
        recorder.start_recording.return_value = Path("data/recordings/out.wav")

        with (
            patch("voicepad.cli.record.get_config", return_value=cfg),
            patch("voicepad.cli.record.AudioRecorder", return_value=recorder),
            patch("voicepad.cli.record._stop_recording", return_value=Path("data/recordings/out.wav")),
            patch("voicepad.cli.record.time.sleep"),
            patch("voicepad.cli.record._transcribe_audio_file"),
        ):
            start_recording(
                prefix=None,
                duration=1.0,
                transcribe=False,
                vad=None,
                min_chunk_duration=None,
                vad_threshold=None,
            )

        recorder.start_recording.assert_called_once()

    def test_start_recording_handles_audio_recorder_error(self) -> None:
        with (
            patch("voicepad.cli.record.get_config", side_effect=AudioRecorderError("bad")),
            self.assertRaises(typer.Exit),
        ):
            start_recording(
                prefix=None,
                duration=None,
                transcribe=False,
                vad=None,
                min_chunk_duration=None,
                vad_threshold=None,
            )

    def test_show_info_happy_path_and_error(self) -> None:
        recordings_path = MagicMock()
        recordings_path.exists.return_value = True
        recordings_path.glob.return_value = [Path("a.wav")]

        cfg = SimpleNamespace(
            input_device_index=1,
            recordings_path=recordings_path,
            recording_prefix="recording",
            vad_enabled=True,
            vad_min_chunk_duration=10.0,
            vad_threshold=0.5,
            vad_min_silence_duration_ms=1000,
        )

        with (
            patch("voicepad.cli.record.get_config", return_value=cfg),
            patch("voicepad.cli.record.typer.echo") as mock_echo,
        ):
            show_info()
        self.assertTrue(mock_echo.called)

        with patch("voicepad.cli.record.get_config", side_effect=RuntimeError("x")), self.assertRaises(typer.Exit):
            show_info()


if __name__ == "__main__":
    unittest.main()
