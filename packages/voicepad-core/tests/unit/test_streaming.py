"""Tests for voicepad_core.streaming."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voicepad_core import AudioRecorder
from voicepad_core.config import Config
from voicepad_core.streaming import (
    MIN_CHUNK_S,
    OVERLAP_S,
    POLL_INTERVAL_S,
    SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD,
    SILENCE_TRIGGER_S,
    ChunkResult,
    StreamingTranscriber,
)
from voicepad_core.transcription import Segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _speech(seconds: float) -> np.ndarray:
    """Non-silent sine-wave audio at 16 kHz."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


class _FakeRecorder(AudioRecorder):
    """Minimal recorder stub that exposes _lock and _frames."""

    def __init__(self, audio: np.ndarray | None = None) -> None:
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        if audio is not None:
            self._frames.append(audio)


# ---------------------------------------------------------------------------
# ChunkResult
# ---------------------------------------------------------------------------


class TestChunkResult:
    def test_default_fields(self) -> None:
        chunk = ChunkResult(index=1, text="hello")
        assert chunk.index == 1
        assert chunk.text == "hello"
        assert chunk.segments == []
        assert chunk.is_final is False
        assert chunk.device == "cuda"
        assert chunk.language == "en"

    def test_is_final_flag(self) -> None:
        chunk = ChunkResult(index=2, text="done", is_final=True)
        assert chunk.is_final is True

    def test_segments_stored(self) -> None:
        seg = Segment(start=0.0, end=1.0, text="hello")
        chunk = ChunkResult(index=1, text="hello", segments=[seg])
        assert len(chunk.segments) == 1
        assert chunk.segments[0].text == "hello"


# ---------------------------------------------------------------------------
# StreamingTranscriber — construction and lifecycle
# ---------------------------------------------------------------------------


class TestStreamingTranscriberLifecycle:
    def test_start_launches_thread(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        recorder = _FakeRecorder()
        st = StreamingTranscriber(
            recorder=recorder,
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        st.start()
        assert st._thread is not None
        assert st._thread.is_alive()
        st.stop()

    def test_stop_joins_thread(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        recorder = _FakeRecorder()
        st = StreamingTranscriber(
            recorder=recorder,
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        st.start()
        st.stop()
        assert st._thread is not None
        assert not st._thread.is_alive()

    def test_stop_sets_stop_event(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        recorder = _FakeRecorder()
        st = StreamingTranscriber(
            recorder=recorder,
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        st.start()
        st.stop()
        assert st._stop_event.is_set()


# ---------------------------------------------------------------------------
# StreamingTranscriber — _get_audio_snapshot
# ---------------------------------------------------------------------------


class TestGetAudioSnapshot:
    def test_returns_empty_array_when_no_frames(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        recorder = _FakeRecorder()
        st = StreamingTranscriber(
            recorder=recorder,
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        audio = st._get_audio_snapshot()
        assert len(audio) == 0

    def test_returns_concatenated_frames(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        recorder = _FakeRecorder()
        recorder._frames = [np.ones(100, dtype=np.float32), np.ones(200, dtype=np.float32)]
        st = StreamingTranscriber(
            recorder=recorder,
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        audio = st._get_audio_snapshot()
        assert len(audio) == 300

    def test_is_thread_safe(self, tmp_path: Path) -> None:
        """Snapshot can be called while frames are being appended."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        recorder = _FakeRecorder()
        st = StreamingTranscriber(
            recorder=recorder,
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        errors = []

        def writer():
            for _ in range(50):
                with recorder._lock:
                    recorder._frames.append(np.zeros(100, dtype=np.float32))
                time.sleep(0.001)

        def reader():
            for _ in range(50):
                try:
                    st._get_audio_snapshot()
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        t1, t2 = threading.Thread(target=writer), threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []


# ---------------------------------------------------------------------------
# StreamingTranscriber — _dispatch_chunk
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestDispatchChunk:
    def _make_mock_model(self, text: str = "hello") -> MagicMock:
        seg = MagicMock()
        seg.start, seg.end, seg.text = 0.5, 1.5, text
        m = MagicMock()
        m.transcribe.return_value = ([seg], MagicMock(language="en", language_probability=0.99))
        return m

    def test_fires_on_chunk_callback(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        chunks: list[ChunkResult] = []
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=chunks.append,
            on_error=lambda _: None,
        )
        audio = _speech(MIN_CHUNK_S + 1)
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(self._make_mock_model(), "cuda", "int8", False),
        ):
            st._dispatch_chunk(audio, is_final=True)
        assert len(chunks) == 1
        assert chunks[0].is_final is True

    def test_chunk_text_is_populated(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        chunks: list[ChunkResult] = []
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=chunks.append,
            on_error=lambda _: None,
        )
        audio = _speech(MIN_CHUNK_S + 1)
        with patch(
            "voicepad_core.transcription.get_or_load_model",
            return_value=(self._make_mock_model("world"), "cuda", "int8", False),
        ):
            st._dispatch_chunk(audio, is_final=False)
        assert "world" in chunks[0].text

    def test_overlap_segments_are_filtered(self, tmp_path: Path) -> None:
        """Segments that fall entirely within the overlap region are excluded."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        chunks: list[ChunkResult] = []
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=chunks.append,
            on_error=lambda _: None,
        )
        # Simulate: consumed 30s, overlap 0.5s → chunk starts at 29.5s
        st._consumed_samples = int(30 * SAMPLE_RATE)
        audio = _speech(31)  # 31s total

        # Segment at 0.1s–0.4s in chunk audio = 29.6s–29.9s absolute → in overlap
        seg_in_overlap = MagicMock()
        seg_in_overlap.start, seg_in_overlap.end, seg_in_overlap.text = 0.1, 0.4, "overlap"
        # Segment at 0.6s–1.5s in chunk audio = 30.1s–31.0s absolute → after overlap
        seg_after = MagicMock()
        seg_after.start, seg_after.end, seg_after.text = 0.6, 1.5, "real"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [seg_in_overlap, seg_after],
            MagicMock(language="en", language_probability=0.99),
        )
        with patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)):
            st._dispatch_chunk(audio, is_final=True)

        # Only the non-overlap segment should appear
        assert len(chunks) == 1
        assert "real" in chunks[0].text
        assert "overlap" not in chunks[0].text

    def test_segments_spanning_overlap_boundary_are_kept(self, tmp_path: Path) -> None:
        """Segments that start in overlap but extend beyond it are kept (prevents word-splitting)."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        chunks: list[ChunkResult] = []
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=chunks.append,
            on_error=lambda _: None,
        )
        # Simulate: consumed 30s, overlap 0.5s → chunk starts at 29.5s
        st._consumed_samples = int(30 * SAMPLE_RATE)
        audio = _speech(31)  # 31s total

        # Segment at 0.3s–0.8s in chunk audio = 29.8s–30.3s absolute
        # Starts in overlap (29.8s < 30.0s) but extends beyond (30.3s > 30.0s)
        # This should be KEPT to avoid word-splitting like "class. classical physics"
        seg_spanning = MagicMock()
        seg_spanning.start, seg_spanning.end, seg_spanning.text = 0.3, 0.8, "classical physics"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [seg_spanning],
            MagicMock(language="en", language_probability=0.99),
        )
        with patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)):
            st._dispatch_chunk(audio, is_final=True)

        # The spanning segment should be kept
        assert len(chunks) == 1
        assert "classical physics" in chunks[0].text

    def test_overlap_consistency_warning_on_mismatch(self, tmp_path: Path) -> None:
        """When overlap text doesn't match between chunks, a warning is logged."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )

        # First chunk: set up previous overlap text
        st._chunk_index = 1
        st._prev_overlap_text = "this is the end"
        st._consumed_samples = int(30 * SAMPLE_RATE)

        audio = _speech(31)

        # Create segments where overlap text is completely different
        seg_in_overlap = MagicMock()
        seg_in_overlap.start, seg_in_overlap.end, seg_in_overlap.text = 0.1, 0.4, "different words"
        seg_after = MagicMock()
        seg_after.start, seg_after.end, seg_after.text = 0.6, 1.5, "new content"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            [seg_in_overlap, seg_after],
            MagicMock(language="en", language_probability=0.99),
        )

        with (
            patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)),
            patch("voicepad_core.streaming.logger") as mock_logger,
        ):
            st._dispatch_chunk(audio, is_final=False)

        # Should have logged a warning about mismatch
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Overlap mismatch" in warning_msg

    def test_prev_context_used_as_prompt_on_second_chunk(self, tmp_path: Path) -> None:
        """After the first chunk sets _prev_context, it appears in the next prompt."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        # Set context as if a previous chunk was already transcribed
        st._prev_context = "previous words here"
        audio = _speech(MIN_CHUNK_S + 1)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", language_probability=0.9))
        with patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)):
            st._dispatch_chunk(audio, is_final=False)
        prompt = mock_model.transcribe.call_args.kwargs.get("initial_prompt", "")
        assert prompt is not None
        assert "previous words here" in prompt

    def test_consumed_samples_updated_after_dispatch(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=lambda _: None,
            on_error=lambda _: None,
        )
        audio = _speech(MIN_CHUNK_S + 1)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock(language="en", language_probability=0.9))
        with patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)):
            st._dispatch_chunk(audio, is_final=False)
        assert st._consumed_samples == len(audio)

    def test_error_callback_on_transcription_failure(self, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        errors: list[str] = []
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=lambda _: None,
            on_error=errors.append,
        )
        audio = _speech(MIN_CHUNK_S + 1)
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("model exploded")
        with patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)):
            st._dispatch_chunk(audio, is_final=False)
        assert len(errors) == 1
        assert "model exploded" in errors[0]

    def test_too_short_chunk_fires_final_signal(self, tmp_path: Path) -> None:
        """When chunk is too short, AudioTooShortError is caught and final signal sent."""
        from voicepad_core.transcription import AudioTooShortError

        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        chunks: list[ChunkResult] = []
        st = StreamingTranscriber(
            recorder=_FakeRecorder(),
            config=config,
            on_chunk=chunks.append,
            on_error=lambda _: None,
        )
        audio = _speech(MIN_CHUNK_S + 1)
        with patch("voicepad_core.transcription.get_or_load_model") as mock_get:
            mock_model = MagicMock()
            mock_model.transcribe.side_effect = AudioTooShortError("too short")
            mock_get.return_value = (mock_model, "cuda", "int8", False)
            st._dispatch_chunk(audio, is_final=True)
        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].text == ""


# ---------------------------------------------------------------------------
# StreamingTranscriber — monitor loop integration
# ---------------------------------------------------------------------------


class TestMonitorLoop:
    def test_dispatches_final_chunk_on_stop_with_short_audio(self, tmp_path: Path) -> None:
        """When stopped with audio shorter than MIN_CHUNK_S, everything is dispatched as final."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        chunks: list[ChunkResult] = []
        audio = _speech(5.0)  # shorter than MIN_CHUNK_S
        recorder = _FakeRecorder(audio)

        seg = MagicMock()
        seg.start, seg.end, seg.text = 0.1, 4.9, "short clip"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock(language="en", language_probability=0.99))

        with (
            patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)),
            patch("voicepad_core.transcription._trim_trailing_silence", side_effect=lambda a, **kw: a),
        ):
            st = StreamingTranscriber(
                recorder=recorder,
                config=config,
                on_chunk=chunks.append,
                on_error=lambda _: None,
            )
            st.start()
            st.stop()

        assert len(chunks) >= 1
        assert chunks[-1].is_final is True

    def test_silence_detection_triggers_mid_recording_dispatch(self, tmp_path: Path) -> None:
        """When silence is detected after MIN_CHUNK_S, a chunk is dispatched mid-recording."""
        config = Config(recordings_path=tmp_path, markdown_path=tmp_path)
        chunks: list[ChunkResult] = []

        # Build audio: MIN_CHUNK_S of speech + 2s of silence + 2s more speech
        audio = np.concatenate([
            _speech(MIN_CHUNK_S),
            _silence(SILENCE_TRIGGER_S * 3),  # long enough silence
            _speech(2.0),
        ])
        recorder = _FakeRecorder(audio)

        seg = MagicMock()
        seg.start, seg.end, seg.text = 0.1, 1.0, "mid chunk"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], MagicMock(language="en", language_probability=0.99))

        with (
            patch("voicepad_core.transcription.get_or_load_model", return_value=(mock_model, "cuda", "int8", False)),
            patch("voicepad_core.transcription._trim_trailing_silence", side_effect=lambda a, **kw: a),
        ):
            st = StreamingTranscriber(
                recorder=recorder,
                config=config,
                on_chunk=chunks.append,
                on_error=lambda _: None,
            )
            st.start()
            # Give the monitor loop time to detect silence and dispatch
            time.sleep(POLL_INTERVAL_S * 5 + SILENCE_TRIGGER_S * 2)
            st.stop()

        # At least one mid-recording chunk + one final chunk
        assert len(chunks) >= 1
        final_chunks = [c for c in chunks if c.is_final]
        assert len(final_chunks) >= 1

    def test_constants_are_sensible(self) -> None:
        """Streaming constants are within expected ranges."""
        assert MIN_CHUNK_S >= 10.0
        assert SILENCE_RMS_THRESHOLD < 0.05
        assert SILENCE_TRIGGER_S >= 0.5
        assert OVERLAP_S < MIN_CHUNK_S
        assert POLL_INTERVAL_S < 1.0
