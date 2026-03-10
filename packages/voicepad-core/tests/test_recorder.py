"""Package-level tests for AudioRecorder behavior."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicepad_core.config import Config
from voicepad_core.recorder import AudioRecorder


class _NeverEndingThread:
    """Test double that always appears alive and never joins."""

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        return


def _create_config(recordings_dir: Path, markdown_dir: Path) -> MagicMock:
    config = MagicMock(spec=Config)
    config.recordings_path = recordings_dir
    config.markdown_path = markdown_dir
    config.recording_prefix = "recording"
    config.vad_enabled = True
    config.vad_min_chunk_duration = 10.0
    config.vad_threshold = 0.5
    config.vad_min_silence_duration_ms = 1000
    config.vad_speech_pad_ms = 400
    config.input_device_index = 1
    return config


def _create_markdown_file(markdown_dir: Path, name: str = "test") -> Path:
    markdown_file = markdown_dir / f"{name}.md"
    markdown_file.write_text(
        "# Test Recording\n\n**Status:** Recording in progress...\n\n---\n\n",
        encoding="utf-8",
    )
    return markdown_file


class RecorderTests(unittest.TestCase):
    def test_stop_recording_saves_raw_audio_without_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            recordings_dir = tmpdir_path / "recordings"
            markdown_dir = tmpdir_path / "markdown"
            recordings_dir.mkdir()
            markdown_dir.mkdir()

            recorder = AudioRecorder(_create_config(recordings_dir, markdown_dir))
            expected_output = recordings_dir / "test.wav"
            recorder._recording = True
            recorder._chunker = object()
            recorder._output_file = expected_output
            recorder._markdown_file = _create_markdown_file(markdown_dir)
            recorder._recorded_frames = [np.zeros((16000, 1), dtype=np.float32)]

            output_file = recorder.stop_recording()

            self.assertEqual(output_file, expected_output)
            self.assertTrue(expected_output.exists())
            self.assertTrue(recorder.wait_for_finalization(timeout=10.0))
            self.assertEqual(recorder.get_last_transcription_state(), "unavailable")

            markdown_content = recorder.get_markdown_path().read_text(encoding="utf-8")  # type: ignore[union-attr]
            self.assertIn("Recording saved; transcription unavailable", markdown_content)

    def test_stop_recording_without_audio_sets_failed_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            recordings_dir = tmpdir_path / "recordings"
            markdown_dir = tmpdir_path / "markdown"
            recordings_dir.mkdir()
            markdown_dir.mkdir()

            recorder = AudioRecorder(_create_config(recordings_dir, markdown_dir))
            expected_output = recordings_dir / "missing.wav"
            recorder._recording = True
            recorder._chunker = object()
            recorder._output_file = expected_output
            recorder._markdown_file = _create_markdown_file(markdown_dir)

            output_file = recorder.stop_recording()

            self.assertIsNone(output_file)
            self.assertFalse(expected_output.exists())
            self.assertTrue(recorder.wait_for_finalization(timeout=10.0))
            self.assertEqual(recorder.get_last_transcription_state(), "failed")

            markdown_content = recorder.get_markdown_path().read_text(encoding="utf-8")  # type: ignore[union-attr]
            self.assertIn("Recording failed", markdown_content)

    def test_stop_recording_returns_quickly_with_busy_chunk_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            recordings_dir = tmpdir_path / "recordings"
            markdown_dir = tmpdir_path / "markdown"
            recordings_dir.mkdir()
            markdown_dir.mkdir()

            recorder = AudioRecorder(_create_config(recordings_dir, markdown_dir))
            expected_output = recordings_dir / "test.wav"
            recorder._recording = True
            recorder._chunker = MagicMock()
            recorder._output_file = expected_output
            recorder._markdown_file = _create_markdown_file(markdown_dir)
            recorder._recorded_frames = [np.zeros((16000, 1), dtype=np.float32)]

            recorder._chunk_worker_running = True
            recorder._chunk_worker_thread = threading.Thread(target=time.sleep, args=(2.0,), daemon=True)
            recorder._chunk_worker_thread.start()

            start_time = time.monotonic()
            output_file = recorder.stop_recording()
            elapsed = time.monotonic() - start_time

            self.assertEqual(output_file, expected_output)
            self.assertLess(elapsed, 1.0)
            self.assertTrue(recorder.wait_for_finalization(timeout=10.0))

    def test_finalize_worker_marks_incomplete_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            recordings_dir = tmpdir_path / "recordings"
            markdown_dir = tmpdir_path / "markdown"
            recordings_dir.mkdir()
            markdown_dir.mkdir()

            recorder = AudioRecorder(_create_config(recordings_dir, markdown_dir))
            output_file = recordings_dir / "test.wav"
            output_file.write_bytes(b"RIFF")

            recorder._chunker = MagicMock()
            recorder._chunker.add_audio.return_value = None
            recorder._chunk_worker_running = True
            recorder._chunk_worker_thread = _NeverEndingThread()  # type: ignore[assignment]
            recorder._markdown_file = _create_markdown_file(markdown_dir)
            recorder._finalized_output_file = output_file
            recorder._audio_queue.put(np.zeros((100, 1), dtype=np.float32))

            recorder._finalize_worker()

            self.assertEqual(recorder.get_last_transcription_state(), "incomplete")
            markdown_content = recorder.get_markdown_path().read_text(encoding="utf-8")  # type: ignore[union-attr]
            self.assertIn("Recording saved; transcription incomplete", markdown_content)


if __name__ == "__main__":
    unittest.main()
