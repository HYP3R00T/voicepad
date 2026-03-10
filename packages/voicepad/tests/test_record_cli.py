"""Package-level tests for record CLI behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voicepad-core" / "src"))

from voicepad.cli.record import _stop_recording
from voicepad_core.recorder import AudioRecorderError


class RecordCliTests(unittest.TestCase):
    def test_stop_reports_error_when_output_file_missing(self) -> None:
        fake_recorder = MagicMock()
        fake_recorder.stop_recording.return_value = Path("D:/voicepad/data/recordings/missing.wav")

        with patch("voicepad.cli.record.typer.secho") as mock_secho:
            output_file = _stop_recording(fake_recorder)

        self.assertIsNone(output_file)
        self.assertTrue(
            any("no audio file was saved" in str(call.args[0]).lower() for call in mock_secho.call_args_list)
        )

    def test_stop_reports_background_transcription_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_file = tmpdir_path / "recordings" / "ok.wav"
            markdown_file = tmpdir_path / "markdown" / "ok.md"
            output_file.parent.mkdir(parents=True)
            markdown_file.parent.mkdir(parents=True)
            output_file.write_bytes(b"RIFF")
            markdown_file.write_text("# test", encoding="utf-8")

            fake_recorder = MagicMock()
            fake_recorder.stop_recording.return_value = output_file
            fake_recorder.get_markdown_path.return_value = markdown_file

            with (
                patch("voicepad.cli.record.typer.secho") as mock_secho,
                patch("voicepad.cli.record.typer.echo") as mock_echo,
            ):
                result = _stop_recording(fake_recorder)

        self.assertEqual(result, output_file)
        self.assertTrue(
            any(
                "transcription ongoing in the background" in str(call.args[0]).lower()
                for call in mock_secho.call_args_list
            )
        )
        self.assertTrue(any(str(markdown_file) in str(call.args[0]) for call in mock_echo.call_args_list if call.args))

    def test_stop_handles_audio_recorder_error(self) -> None:
        fake_recorder = MagicMock()
        fake_recorder.stop_recording.side_effect = AudioRecorderError("boom")

        with patch("voicepad.cli.record.typer.secho") as mock_secho:
            output_file = _stop_recording(fake_recorder)

        self.assertIsNone(output_file)
        self.assertTrue(
            any("error stopping recording" in str(call.args[0]).lower() for call in mock_secho.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
