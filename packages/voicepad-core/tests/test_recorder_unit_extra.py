"""Additional unit tests for AudioRecorder internals."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicepad_core.chunking import ChunkMetadata
from voicepad_core.config import Config
from voicepad_core.recorder import AudioRecorder, AudioRecorderError


def _create_config(recordings_dir: Path, markdown_dir: Path, *, vad_enabled: bool = False) -> MagicMock:
    config = MagicMock(spec=Config)
    config.recordings_path = recordings_dir
    config.markdown_path = markdown_dir
    config.recording_prefix = "recording"
    config.vad_enabled = vad_enabled
    config.vad_min_chunk_duration = 10.0
    config.vad_threshold = 0.5
    config.vad_min_silence_duration_ms = 1000
    config.vad_speech_pad_ms = 400
    config.input_device_index = 1
    return config


def _make_markdown(markdown_dir: Path, name: str = "test") -> Path:
    markdown_dir.mkdir(parents=True, exist_ok=True)
    path = markdown_dir / f"{name}.md"
    path.write_text("# Header\n\n---\n", encoding="utf-8")
    return path


class RecorderUnitExtraTests(unittest.TestCase):
    def test_generate_filename_and_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recorder = AudioRecorder(_create_config(root / "r", root / "m"))
            filename = recorder._generate_filename()
            self.assertTrue(filename.startswith("recording_"))
            self.assertTrue(filename.endswith(".wav"))
            self.assertEqual(recorder._get_output_path("x").parent, root / "r")

    def test_audio_callback_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recorder = AudioRecorder(_create_config(root / "r", root / "m"))
            recorder._recording = True

            frame = np.ones((10, 1), dtype=np.float32)
            status = cast(Any, SimpleNamespace())
            recorder._audio_callback(frame, 10, {}, status)
            self.assertFalse(recorder._audio_queue.empty())

            recorder._chunker = object()
            recorder._audio_callback(frame, 10, {}, status)
            self.assertGreaterEqual(len(recorder._recorded_frames), 1)

    def test_update_markdown_status_and_finalize_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recorder = AudioRecorder(_create_config(root / "r", root / "m"))
            markdown = _make_markdown(root / "m")
            recorder._markdown_file = markdown

            recorder._update_markdown_status("Recording complete", note="done")
            content = markdown.read_text(encoding="utf-8")
            self.assertIn("**Status:** Recording complete", content)
            self.assertIn("> Note: done", content)

            recorder._finalize_markdown_status(audio_saved=False, worker_timed_out=False, queued_frames_remaining=0)
            self.assertEqual(recorder.get_last_transcription_state(), "failed")

            recorder._finalize_markdown_status(audio_saved=True, worker_timed_out=True, queued_frames_remaining=1)
            self.assertEqual(recorder.get_last_transcription_state(), "incomplete")

            recorder._accumulated_chunks = [np.zeros((10,), dtype=np.float32)]
            recorder._finalize_markdown_status(audio_saved=True, worker_timed_out=False, queued_frames_remaining=0)
            self.assertEqual(recorder.get_last_transcription_state(), "complete")

            recorder._accumulated_chunks = []
            recorder._finalize_markdown_status(audio_saved=True, worker_timed_out=False, queued_frames_remaining=0)
            self.assertEqual(recorder.get_last_transcription_state(), "unavailable")

    def test_start_recording_vad_initializes_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recordings = root / "recordings"
            markdown = root / "markdown"
            recordings.mkdir()
            markdown.mkdir()

            recorder = AudioRecorder(_create_config(recordings, markdown, vad_enabled=True))

            fake_chunker = MagicMock()
            fake_chunker._get_vad_model.return_value = object()
            fake_thread = MagicMock()
            fake_thread.is_alive.return_value = False

            with (
                patch("voicepad_core.chunking.RealtimeChunker", return_value=fake_chunker),
                patch("voicepad_core.recorder.threading.Thread", return_value=fake_thread),
            ):
                out = recorder.start_recording(prefix="test", duration=1.0)

            self.assertTrue(str(out).endswith(".wav"))
            self.assertTrue((markdown / f"{out.stem}.md").exists())

    def test_chunk_worker_processes_queue_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recorder = AudioRecorder(_create_config(root / "r", root / "m"))
            frame = np.ones((10, 1), dtype=np.float32)
            metadata = ChunkMetadata(index=0, start_time=0.0, duration=0.1, sample_count=10)

            chunker = MagicMock()
            chunker.add_audio.return_value = (frame.flatten(), metadata)
            chunker.finalize.return_value = None
            recorder._chunker = chunker
            recorder._chunk_worker_running = False
            recorder._audio_queue.put(frame)

            with patch.object(recorder, "_handle_completed_chunk") as mock_handle:
                recorder._chunk_worker()

            self.assertTrue(recorder._chunk_worker_finished)
            self.assertTrue(mock_handle.called)

    def test_transcribe_and_append_chunk_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recordings = root / "recordings"
            markdown = root / "markdown"
            recordings.mkdir()
            markdown.mkdir()

            recorder = AudioRecorder(_create_config(recordings, markdown))
            recorder._markdown_file = _make_markdown(markdown)
            audio = np.zeros((160,), dtype=np.float32)
            meta = ChunkMetadata(index=0, start_time=0.0, duration=0.1, sample_count=160)

            with patch("voicepad_core.transcription.transcribe_chunk_to_markdown", return_value="## Chunk 1\n\ntext\n"):
                recorder._transcribe_and_append_chunk(audio, meta)

            content = recorder._markdown_file.read_text(encoding="utf-8")
            self.assertIn("Chunk 1", content)

            with patch("voicepad_core.recorder.sf.write", side_effect=OSError("disk")):
                recorder._transcribe_and_append_chunk(audio, meta)

    def test_record_worker_non_vad_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recordings = root / "recordings"
            markdown = root / "markdown"
            recordings.mkdir()
            markdown.mkdir()

            recorder = AudioRecorder(_create_config(recordings, markdown))
            recorder._output_file = recordings / "out.wav"
            recorder._recording = False
            recorder._audio_queue.put(np.zeros((10, 1), dtype=np.float32))

            stream_cm = MagicMock()
            stream_cm.__enter__.return_value = stream_cm
            stream_cm.__exit__.return_value = None

            with (
                patch("voicepad_core.recorder.sd.InputStream", return_value=stream_cm),
                patch.object(recorder, "_save_recording") as mock_save,
            ):
                recorder._record_worker(duration=None)
            self.assertTrue(mock_save.called)

            with (
                patch("voicepad_core.recorder.sd.InputStream", side_effect=RuntimeError("device")),
                self.assertRaises(AudioRecorderError),
            ):
                recorder._record_worker(duration=None)

    def test_save_helpers_and_wait_for_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recordings = root / "recordings"
            markdown = root / "markdown"
            recordings.mkdir()
            markdown.mkdir()

            recorder = AudioRecorder(_create_config(recordings, markdown))
            recorder._output_file = recordings / "out.wav"
            recorder._accumulated_chunks = [np.zeros((10,), dtype=np.float32)]
            recorder._save_accumulated_audio()
            self.assertTrue(recorder._output_file.exists())

            recorder._save_recording([np.zeros((10, 1), dtype=np.float32)])
            self.assertTrue(recorder._output_file.exists())

            recorder._finalization_thread = MagicMock()
            recorder._finalization_thread.is_alive.return_value = True
            self.assertFalse(recorder.wait_for_finalization(timeout=0.0))


if __name__ == "__main__":
    unittest.main()
