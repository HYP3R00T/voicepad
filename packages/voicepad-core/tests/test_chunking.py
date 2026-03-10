"""Unit tests for RealtimeChunker."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicepad_core.chunking import RealtimeChunker


class ChunkingTests(unittest.TestCase):
    def test_init_validation(self) -> None:
        with self.assertRaises(ValueError):
            RealtimeChunker(min_chunk_duration=5.0)
        with self.assertRaises(ValueError):
            RealtimeChunker(vad_threshold=2.0)
        with self.assertRaises(ValueError):
            RealtimeChunker(min_silence_duration_ms=50)

    def test_get_vad_model_lazy_load(self) -> None:
        chunker = RealtimeChunker(min_chunk_duration=10.0)
        fake_model = MagicMock()
        with patch("faster_whisper.vad.get_vad_model", return_value=fake_model) as mock_get:
            model1 = chunker._get_vad_model()
            model2 = chunker._get_vad_model()
        self.assertIs(model1, fake_model)
        self.assertIs(model2, fake_model)
        mock_get.assert_called_once_with()

    def test_concatenate_buffer_and_finalize(self) -> None:
        chunker = RealtimeChunker(min_chunk_duration=10.0)
        self.assertEqual(len(chunker._concatenate_buffer()), 0)

        chunker._buffer = [np.ones((16000, 1), dtype=np.float32)]
        chunker._buffer_duration = 1.0
        final_chunk = chunker.finalize()
        self.assertIsNotNone(final_chunk)
        assert final_chunk is not None
        audio, metadata = final_chunk
        self.assertEqual(metadata.index, 0)
        self.assertEqual(len(audio), 16000)

    def test_analyze_buffer_for_boundary(self) -> None:
        chunker = RealtimeChunker(min_chunk_duration=10.0, min_silence_duration_ms=500)
        chunker._buffer = [np.zeros((32000, 1), dtype=np.float32)]

        with patch("faster_whisper.vad.get_speech_timestamps", return_value=[]):
            self.assertIsNone(chunker._analyze_buffer_for_boundary())

        speech = [{"end": 24000}]
        with patch("faster_whisper.vad.get_speech_timestamps", return_value=speech):
            boundary = chunker._analyze_buffer_for_boundary()
        self.assertEqual(boundary, 24000)

    def test_add_audio_below_threshold_then_chunk(self) -> None:
        chunker = RealtimeChunker(min_chunk_duration=10.0, sample_rate=16000)

        first = chunker.add_audio(np.ones((80000, 1), dtype=np.float32))
        self.assertIsNone(first)

        with patch.object(chunker, "_analyze_buffer_for_boundary", return_value=120000):
            second = chunker.add_audio(np.ones((80000, 1), dtype=np.float32))

        self.assertIsNotNone(second)
        assert second is not None
        audio, metadata = second
        self.assertEqual(metadata.index, 0)
        self.assertGreater(len(audio), 0)

    def test_finalize_empty_and_reset(self) -> None:
        chunker = RealtimeChunker(min_chunk_duration=10.0)
        self.assertIsNone(chunker.finalize())

        chunker._buffer = [np.ones((1600, 1), dtype=np.float32)]
        chunker._buffer_duration = 0.1
        chunker._total_processed_duration = 1.0
        chunker._chunk_index = 3

        chunker.reset()
        self.assertEqual(chunker._buffer, [])
        self.assertEqual(chunker._buffer_duration, 0.0)
        self.assertEqual(chunker._total_processed_duration, 0.0)
        self.assertEqual(chunker._chunk_index, 0)


if __name__ == "__main__":
    unittest.main()
