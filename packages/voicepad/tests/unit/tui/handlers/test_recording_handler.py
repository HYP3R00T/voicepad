"""Tests for RecordingHandler."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock, patch

from voicepad.tui.handlers.recording_handler import RecordingHandler
from voicepad_core import ChunkResult, WavArtifact


def _artifact(path: Path, frame_count: int = 2, sample_rate: int = 16_000) -> WavArtifact:
    return WavArtifact(path, sample_rate, 1, frame_count, frame_count / sample_rate)


class TestRecordingHandlerInit:
    """Tests for RecordingHandler initialization."""

    def test_init_stores_app_reference(self):
        """Test that __init__ stores the app reference."""
        mock_app = Mock()
        handler = RecordingHandler(mock_app)
        assert handler.app is mock_app


class TestActionToggleRecording:
    """Tests for action_toggle_recording method."""

    def test_returns_early_when_not_on_record_tab(self):
        """Test that action_toggle_recording returns early when not on record tab."""
        mock_app = Mock()
        mock_tabs = Mock()
        mock_tabs.active = "tab-history"
        mock_app.query_one.return_value = mock_tabs
        mock_app._model_ready = True
        mock_app._transcribing = False
        mock_app._recording = False

        handler = RecordingHandler(mock_app)

        with (
            patch.object(handler, "start_recording") as mock_start,
            patch.object(handler, "stop_recording") as mock_stop,
        ):
            handler.action_toggle_recording()
            mock_start.assert_not_called()
            mock_stop.assert_not_called()

    def test_returns_early_when_model_not_ready(self):
        """Test that action_toggle_recording returns early when model not ready."""
        mock_app = Mock()
        mock_tabs = Mock()
        mock_tabs.active = "tab-record"
        mock_app.query_one.return_value = mock_tabs
        mock_app._model_ready = False
        mock_app._transcribing = False
        mock_app._recording = False

        handler = RecordingHandler(mock_app)

        with (
            patch.object(handler, "start_recording") as mock_start,
            patch.object(handler, "stop_recording") as mock_stop,
        ):
            handler.action_toggle_recording()
            mock_start.assert_not_called()
            mock_stop.assert_not_called()

    def test_returns_early_when_transcribing(self):
        """Test that action_toggle_recording returns early when transcribing."""
        mock_app = Mock()
        mock_tabs = Mock()
        mock_tabs.active = "tab-record"
        mock_app.query_one.return_value = mock_tabs
        mock_app._model_ready = True
        mock_app._transcribing = True
        mock_app._recording = False

        handler = RecordingHandler(mock_app)

        with (
            patch.object(handler, "start_recording") as mock_start,
            patch.object(handler, "stop_recording") as mock_stop,
        ):
            handler.action_toggle_recording()
            mock_start.assert_not_called()
            mock_stop.assert_not_called()

    def test_calls_stop_recording_when_recording(self):
        """Test that action_toggle_recording calls stop_recording when recording."""
        mock_app = Mock()
        mock_tabs = Mock()
        mock_tabs.active = "tab-record"
        mock_app.query_one.return_value = mock_tabs
        mock_app._model_ready = True
        mock_app._transcribing = False
        mock_app._recording = True

        handler = RecordingHandler(mock_app)

        with (
            patch.object(handler, "start_recording") as mock_start,
            patch.object(handler, "stop_recording") as mock_stop,
        ):
            handler.action_toggle_recording()
            mock_start.assert_not_called()
            mock_stop.assert_called_once()

    def test_calls_start_recording_when_not_recording(self):
        """Test that action_toggle_recording calls start_recording when not recording."""
        mock_app = Mock()
        mock_tabs = Mock()
        mock_tabs.active = "tab-record"
        mock_app.query_one.return_value = mock_tabs
        mock_app._model_ready = True
        mock_app._transcribing = False
        mock_app._recording = False

        handler = RecordingHandler(mock_app)

        with (
            patch.object(handler, "start_recording") as mock_start,
            patch.object(handler, "stop_recording") as mock_stop,
        ):
            handler.action_toggle_recording()
            mock_start.assert_called_once()
            mock_stop.assert_not_called()


class TestStartRecording:
    """Tests for start_recording method."""

    @patch("voicepad.tui.handlers.recording_handler.begin_transcription_session")
    @patch("voicepad.tui.handlers.recording_handler.RecordingSession")
    @patch("voicepad.tui.handlers.recording_handler.StreamingTranscriber")
    @patch("voicepad.tui.handlers.recording_handler.time")
    def test_creates_recording_session(self, mock_time, mock_transcriber, mock_session_class, mock_begin_session):
        """Test that start_recording creates a RecordingSession."""
        mock_app = Mock()
        mock_app.config = Mock()
        mock_app.config.logs_path = Path("/tmp/logs")
        mock_session = Mock()
        mock_session._recorder = Mock()
        mock_session_class.return_value = mock_session
        mock_time.monotonic.return_value = 100.0
        mock_begin_session.return_value = (Mock(), Path("/tmp/test.log"))

        handler = RecordingHandler(mock_app)
        handler.start_recording()

        mock_session_class.assert_called_once_with(config=mock_app.config)
        mock_session.start.assert_called_once()

    @patch("voicepad.tui.handlers.recording_handler.begin_transcription_session")
    @patch("voicepad.tui.handlers.recording_handler.RecordingSession")
    def test_handles_audio_recorder_error(self, mock_session_class, mock_begin_session):
        """Test that start_recording handles RuntimeError."""
        mock_app = Mock()
        mock_app.config = Mock()
        mock_app.config.logs_path = Path("/tmp/logs")
        mock_session = Mock()
        mock_session.start.side_effect = RuntimeError("Test error")
        mock_session_class.return_value = mock_session
        mock_begin_session.return_value = (Mock(), Path("/tmp/test.log"))

        handler = RecordingHandler(mock_app)
        handler.start_recording()

        mock_app._set_status.assert_called_once_with("error", "mic error: Test error")

    @patch("voicepad.tui.handlers.recording_handler.begin_transcription_session")
    @patch("voicepad.tui.handlers.recording_handler.RecordingSession")
    @patch("voicepad.tui.handlers.recording_handler.StreamingTranscriber")
    @patch("voicepad.tui.handlers.recording_handler.time")
    def test_sets_recording_state(self, mock_time, mock_transcriber, mock_session_class, mock_begin_session):
        """Test that start_recording sets recording state."""
        mock_app = Mock()
        mock_app.config = Mock()
        mock_app.config.logs_path = Path("/tmp/logs")
        mock_session = Mock()
        mock_session._recorder = Mock()
        mock_session_class.return_value = mock_session
        mock_time.monotonic.return_value = 100.0
        mock_begin_session.return_value = (Mock(), Path("/tmp/test.log"))

        handler = RecordingHandler(mock_app)
        handler.start_recording()

        assert mock_app._recording is True
        assert mock_app._record_start == 100.0
        assert mock_app._stream_chunks == []

    @patch("voicepad.tui.handlers.recording_handler.begin_transcription_session")
    @patch("voicepad.tui.handlers.recording_handler.RecordingSession")
    @patch("voicepad.tui.handlers.recording_handler.StreamingTranscriber")
    @patch("voicepad.tui.handlers.recording_handler.time")
    def test_starts_timer_and_updates_status(self, mock_time, mock_transcriber, mock_session_class, mock_begin_session):
        """Test that start_recording starts timer and updates status."""
        mock_app = Mock()
        mock_app.config = Mock()
        mock_app.config.logs_path = Path("/tmp/logs")
        mock_session = Mock()
        mock_session._recorder = Mock()
        mock_session_class.return_value = mock_session
        mock_time.monotonic.return_value = 100.0
        mock_begin_session.return_value = (Mock(), Path("/tmp/test.log"))

        handler = RecordingHandler(mock_app)
        handler.start_recording()

        mock_app._set_status.assert_called_once_with("recording", "recording…")
        mock_app._start_timer.assert_called_once()

    @patch("voicepad.tui.handlers.recording_handler.begin_transcription_session")
    @patch("voicepad.tui.handlers.recording_handler.RecordingSession")
    @patch("voicepad.tui.handlers.recording_handler.StreamingTranscriber")
    @patch("voicepad.tui.handlers.recording_handler.time")
    def test_starts_streaming_transcriber(
        self, mock_time, mock_transcriber_class, mock_session_class, mock_begin_session
    ):
        """Test that start_recording starts StreamingTranscriber."""
        mock_app = Mock()
        mock_app.config = Mock()
        mock_app.config.logs_path = Path("/tmp/logs")
        mock_app.config.transcription_model = "turbo"
        mock_app.config.transcription_device = "cuda"
        mock_app.config.transcription_compute_type = "int8_float16"
        mock_app.config.min_chunk_s = 1.0
        mock_app.config.max_chunk_s = 30.0
        mock_app.config.overlap_s = 2.0
        mock_app.config.silence_threshold_ms = 500
        mock_session = Mock()
        mock_recorder = Mock()
        mock_session._recorder = mock_recorder
        mock_session_class.return_value = mock_session
        mock_time.monotonic.return_value = 100.0
        mock_transcriber = Mock()
        mock_transcriber_class.return_value = mock_transcriber
        mock_begin_session.return_value = (Mock(), Path("/tmp/test.log"))

        handler = RecordingHandler(mock_app)
        handler.start_recording()

        mock_transcriber_class.assert_called_once()
        call_kwargs = mock_transcriber_class.call_args[1]
        assert call_kwargs["recorder"] is mock_recorder
        assert call_kwargs["model_name"] == "turbo"
        assert call_kwargs["device"] == "cuda"
        assert call_kwargs["compute_type"] == "int8_float16"
        assert call_kwargs["min_chunk_s"] == 1.0
        assert call_kwargs["max_chunk_s"] == 30.0
        assert call_kwargs["overlap_s"] == 2.0
        assert call_kwargs["silence_threshold_ms"] == 500
        assert callable(call_kwargs["on_chunk"])
        assert callable(call_kwargs["on_error"])
        mock_transcriber.start.assert_called_once()

    @patch("voicepad.tui.handlers.recording_handler.begin_transcription_session")
    @patch("voicepad.tui.handlers.recording_handler.RecordingSession")
    @patch("voicepad.tui.handlers.recording_handler.StreamingTranscriber")
    @patch("voicepad.tui.handlers.recording_handler.time")
    def test_uses_core_transcription_session_logging(
        self, mock_time, mock_transcriber_class, mock_session_class, mock_begin_session
    ):
        """start_recording should delegate transcription-session logging setup to voicepad_core."""
        mock_app = Mock()
        mock_app.config = Mock()
        mock_app.config.logs_path = Path("/tmp/logs")
        mock_session = Mock()
        mock_session._recorder = Mock()
        mock_session_class.return_value = mock_session
        mock_time.monotonic.return_value = 100.0
        mock_begin_session.return_value = (Mock(), Path("/tmp/test.log"))
        mock_transcriber_class.return_value = Mock()

        handler = RecordingHandler(mock_app)
        handler.start_recording()

        call_kwargs = mock_begin_session.call_args.kwargs
        assert call_kwargs["logs_path"] == Path("/tmp/logs")
        assert call_kwargs["log_level"] == "INFO"
        assert call_kwargs["include_streaming"] is True


class TestStopRecording:
    """Tests for stop_recording method."""

    def test_returns_early_when_no_session(self):
        """Test that stop_recording returns early when no session."""
        mock_app = Mock()
        mock_app._session = None

        handler = RecordingHandler(mock_app)
        handler.stop_recording()

        # Should not crash and should not call any methods

    @patch.object(RecordingHandler, "finalize_worker")
    def test_stops_recording_and_timer(self, mock_finalize):
        """Test that stop_recording stops recording and timer."""
        mock_app = Mock()
        mock_session = Mock()
        mock_session.stop.return_value = _artifact(Path("/tmp/recording.wav"), 3)
        mock_app._session = mock_session

        handler = RecordingHandler(mock_app)
        handler.stop_recording()

        assert mock_app._recording is False
        mock_app._stop_timer.assert_called_once()
        mock_app._set_status.assert_called_with("transcribing", "transcribing…")

    @patch.object(RecordingHandler, "finalize_worker")
    def test_handles_audio_recorder_error_on_stop(self, mock_finalize):
        """Test that stop_recording handles RuntimeError."""
        mock_app = Mock()
        mock_session = Mock()
        mock_session.stop.side_effect = RuntimeError("Stop error")
        mock_app._session = mock_session
        mock_app._streamer = Mock()

        handler = RecordingHandler(mock_app)
        handler.stop_recording()

        mock_app._set_status.assert_called_with("error", "stop error: Stop error")
        mock_app._streamer._stop_event.set.assert_called_once()
        mock_finalize.assert_not_called()

    @patch.object(RecordingHandler, "finalize_worker")
    def test_sets_transcribing_state_and_calls_finalize(self, mock_finalize):
        """Test that stop_recording sets transcribing state and delegates to app._finalize_worker."""
        mock_app = Mock()
        mock_session = Mock()
        audio = _artifact(Path("/tmp/recording.wav"), 3)
        mock_session.stop.return_value = audio
        mock_app._session = mock_session

        handler = RecordingHandler(mock_app)
        handler.stop_recording()

        assert mock_app._transcribing is True
        mock_app._finalize_worker.assert_called_once()
        # Check that audio array was passed
        call_args = mock_app._finalize_worker.call_args[0]
        assert call_args[0] is audio


class TestFinalizeWorker:
    """Tests for finalize_worker and final full-audio fallback."""

    @patch("voicepad.tui.handlers.recording_handler.end_transcription_session")
    @patch.object(RecordingHandler, "_transcribe_final_audio")
    def test_finalize_worker_uses_final_full_audio_result_for_simple_sessions(self, mock_final_pass, mock_end_session):
        mock_app = Mock()
        mock_app._streamer = Mock()
        mock_app._stream_chunks = []
        mock_app.call_from_thread = Mock()

        handler = RecordingHandler(mock_app)
        handler._final_chunk_event = threading.Event()
        handler._final_chunk_event.set()
        handler._session_logger = Mock()
        handler._log_file = Path("/tmp/test.log")

        audio = _artifact(Path("/tmp/recording.wav"), 3)
        final_result = Mock()
        mock_final_pass.return_value = final_result

        handler.finalize_worker(audio)

        mock_app._streamer.stop.assert_called_once_with(transcribe_tail=False)
        mock_final_pass.assert_called_once()
        assert mock_final_pass.call_args[0][0] is audio
        mock_app.call_from_thread.assert_called_once_with(handler._save_final_pass, audio, final_result)
        mock_end_session.assert_called_once_with(include_streaming=True)

    @patch("voicepad.tui.handlers.recording_handler.end_transcription_session")
    @patch.object(RecordingHandler, "_transcribe_final_audio")
    def test_finalize_worker_drains_tail_without_second_pass_when_streaming_has_text(
        self,
        mock_final_pass,
        mock_end_session,
    ):
        """An existing streamed result drains its tail and skips duplicate full-audio inference."""
        mock_app = Mock()
        mock_app._streamer = Mock()
        mock_app._stream_chunks = [Mock()]
        mock_app.call_from_thread = Mock()

        handler = RecordingHandler(mock_app)
        handler._final_chunk_event = threading.Event()
        handler._final_chunk_event.set()
        audio = _artifact(Path("/tmp/recording.wav"), 3)

        handler.finalize_worker(audio)

        mock_app._streamer.stop.assert_called_once_with(transcribe_tail=True)
        mock_final_pass.assert_not_called()
        mock_app.call_from_thread.assert_called_once_with(handler.save_recording, audio, None)
        mock_end_session.assert_called_once_with(include_streaming=True)


class TestOnStreamChunk:
    """Tests for on_stream_chunk method."""

    def test_appends_chunk_with_text(self):
        """Test that on_stream_chunk appends chunk with text."""
        mock_app = Mock()
        mock_app._stream_chunks = []
        mock_label = Mock()
        mock_static = Mock()
        mock_app.query_one.side_effect = [mock_label, mock_static]

        chunk = ChunkResult(text="Hello", is_final=False, device="cuda", latency_ms=100.0, segments=[], index=0)

        handler = RecordingHandler(mock_app)
        handler.on_stream_chunk(chunk)

        assert len(mock_app._stream_chunks) == 1
        assert mock_app._stream_chunks[0] is chunk

    def test_updates_transcription_display(self):
        """Test that on_stream_chunk updates transcription display."""
        mock_app = Mock()
        chunk1 = ChunkResult(text="Hello", is_final=False, device="cuda", latency_ms=100.0, segments=[], index=0)
        chunk2 = ChunkResult(text="world", is_final=False, device="cuda", latency_ms=100.0, segments=[], index=1)
        mock_app._stream_chunks = [chunk1]
        mock_label = Mock()
        mock_static = Mock()
        mock_app.query_one.side_effect = [mock_label, mock_static]

        handler = RecordingHandler(mock_app)
        handler.on_stream_chunk(chunk2)

        mock_label.remove_class.assert_called_once_with("placeholder")
        mock_label.update.assert_called_once_with("Hello world")
        mock_static.scroll_end.assert_called_once_with(animate=False)

    @patch("voicepad.tui.handlers.recording_handler.time")
    def test_handles_final_chunk(self, mock_time):
        """Test that on_stream_chunk handles final chunk."""
        mock_app = Mock()
        mock_app._record_start = 100.0
        chunk = ChunkResult(text="Final", is_final=True, device="cuda", latency_ms=100.0, segments=[], index=0)
        mock_app._stream_chunks = [chunk]
        mock_app._hotkey_pending_copy = False
        mock_time.monotonic.return_value = 105.5
        mock_label = Mock()
        mock_static = Mock()
        mock_meta_label = Mock()
        mock_app.query_one.side_effect = [mock_label, mock_static, mock_meta_label]

        handler = RecordingHandler(mock_app)
        handler.on_stream_chunk(chunk)

        assert mock_app._transcribing is False
        mock_meta_label.update.assert_called_once_with("[dim]5.5s  ·  streaming[/]")
        mock_app._set_status.assert_called_with("ready", "ready")
        mock_app._overlay_set.assert_called_once_with("hidden")

    @patch("voicepad.tui.handlers.recording_handler.time")
    @patch("voicepad.tui.handlers.recording_handler._copy_to_clipboard")
    def test_auto_copies_on_hotkey_trigger(self, mock_copy, mock_time):
        """Test that on_stream_chunk auto-copies when triggered by hotkey."""
        mock_app = Mock()
        mock_app._record_start = 100.0
        chunk = ChunkResult(text="Copy me", is_final=True, device="cuda", latency_ms=100.0, segments=[], index=0)
        mock_app._stream_chunks = []  # Start empty, chunk will be added by on_stream_chunk
        mock_app._hotkey_pending_copy = True
        mock_time.monotonic.return_value = 105.0
        mock_label = Mock()
        mock_static = Mock()
        mock_meta_label = Mock()
        mock_app.query_one.side_effect = [mock_label, mock_static, mock_meta_label]

        handler = RecordingHandler(mock_app)
        handler.on_stream_chunk(chunk)

        assert mock_app._hotkey_pending_copy is False
        mock_copy.assert_called_once_with("Copy me")
        mock_app._set_status.assert_called_with("ready", "ready — copied to clipboard")
        mock_app._overlay_set.assert_called_with("copied")

    def test_final_chunk_signals_completion_event(self):
        """Test that the final streamed chunk signals stop completion."""
        mock_app = Mock()
        mock_app._stream_chunks = []
        mock_app._record_start = 0.0
        mock_app._hotkey_pending_copy = False
        mock_label = Mock()
        mock_static = Mock()
        mock_meta_label = Mock()
        mock_app.query_one.side_effect = [mock_label, mock_static, mock_meta_label]

        handler = RecordingHandler(mock_app)
        handler._final_chunk_event = threading.Event()

        chunk = ChunkResult(text="done", is_final=True, device="cuda", latency_ms=100.0, segments=[], index=0)
        handler._handle_stream_chunk(chunk)

        assert handler._final_chunk_event.is_set()

    def test_stream_error_clears_transcribing_and_signals_completion(self):
        """Test that a streaming error clears transcribing state and unblocks stop."""
        mock_app = Mock()
        mock_app._transcribing = True

        handler = RecordingHandler(mock_app)
        handler._final_chunk_event = threading.Event()

        handler._handle_stream_error("chunk failed")

        assert mock_app._transcribing is False
        mock_app._set_status.assert_called_once_with("error", "chunk failed")
        assert handler._final_chunk_event.is_set()


class TestSaveRecording:
    """Tests for save_recording method."""

    def test_returns_early_when_no_text(self):
        """Test that save_recording still handles empty text without crashing."""
        mock_app = Mock()
        mock_app._stream_chunks = []
        mock_app._entries = []
        mock_app.config.recordings_path = Path("/tmp/recordings")
        mock_app.config.markdown_path = Path("/tmp/markdown")
        mock_app.config.recording_prefix = "recording"

        mock_button = Mock()
        mock_app.query_one.return_value = mock_button

        wav_path = Path("/tmp/recordings/recording_20220101_120000.wav")
        audio = _artifact(wav_path)

        handler = RecordingHandler(mock_app)
        with (
            patch("voicepad.tui.handlers.recording_handler.time.strftime", return_value="20220101_120000"),
            patch.object(Path, "write_text"),
            patch.object(Path, "mkdir"),
        ):
            handler.save_recording(audio)

        assert mock_app._current_text == ""
        assert mock_button.disabled is True
        assert len(mock_app._entries) == 1

    @patch("voicepad.tui.handlers.recording_handler._format_markdown_streaming")
    def test_saves_wav_even_when_transcript_is_empty(self, mock_format_md):
        """Test that save_recording still saves WAV output when transcript text is empty."""
        mock_app = Mock()
        mock_app._stream_chunks = []
        mock_app._entries = []
        mock_app.config.recordings_path = Path("/tmp/recordings")
        mock_app.config.markdown_path = Path("/tmp/markdown")
        mock_app.config.recording_prefix = "recording"

        mock_button = Mock()
        mock_app.query_one.return_value = mock_button

        wav_path = Path("/tmp/recordings/recording_20220101_120000.wav")
        audio = _artifact(wav_path)

        handler = RecordingHandler(mock_app)
        with (
            patch("voicepad.tui.handlers.recording_handler.time.strftime", return_value="20220101_120000"),
            patch.object(Path, "write_text"),
            patch.object(Path, "mkdir"),
        ):
            handler.save_recording(audio)

        mock_format_md.assert_not_called()
        assert mock_app._current_text == ""
        assert mock_button.disabled is True
        assert len(mock_app._entries) == 1
        entry = mock_app._entries[0]
        assert entry.text == ""
        assert entry.wav_path is not None
        assert entry.md_path is None

    @patch("voicepad.tui.handlers.recording_handler._format_markdown_streaming")
    def test_saves_wav_and_markdown(self, mock_format_md):
        """Test that save_recording saves WAV and markdown files."""
        mock_app = Mock()
        chunk = ChunkResult(text="Test text", is_final=True, device="cuda", latency_ms=100.0, segments=[], index=0)
        mock_app._stream_chunks = [chunk]
        mock_app._entries = []
        mock_app.config.recordings_path = Path("/tmp/recordings")
        mock_app.config.markdown_path = Path("/tmp/markdown")
        mock_app.config.recording_prefix = "recording"

        mock_button = Mock()
        mock_app.query_one.return_value = mock_button

        mock_format_md.return_value = "# Markdown content"

        wav_path = Path("/tmp/recordings/recording_20220101_120000.wav")
        audio = _artifact(wav_path)

        handler = RecordingHandler(mock_app)
        with (
            patch("voicepad.tui.handlers.recording_handler.time.strftime", return_value="20220101_120000"),
            patch.object(Path, "write_text"),
            patch.object(Path, "mkdir"),
        ):
            handler.save_recording(audio)

        assert mock_app._current_text == "Test text"
        assert mock_button.disabled is False

    @patch("voicepad.tui.handlers.recording_handler._format_markdown")
    @patch("voicepad.tui.handlers.recording_handler._format_markdown_streaming")
    def test_save_recording_prefers_final_result_text(self, mock_format_streaming, mock_format_markdown):
        mock_app = Mock()
        chunk = ChunkResult(text="Truncated", is_final=True, device="cuda", latency_ms=100.0, segments=[], index=0)
        mock_app._stream_chunks = [chunk]
        mock_app._entries = []
        mock_app.config.recordings_path = Path("/tmp/recordings")
        mock_app.config.markdown_path = Path("/tmp/markdown")
        mock_app.config.recording_prefix = "recording"
        mock_app.config.transcription_model = "turbo"

        mock_button = Mock()
        mock_app.query_one.return_value = mock_button

        mock_format_markdown.return_value = "# Final markdown"
        final_result = Mock(text="Complete ending at the moment.", latency_ms=321.0, device="cpu")
        wav_path = Path("/tmp/recordings/recording_20220101_120000.wav")
        audio = _artifact(wav_path)

        handler = RecordingHandler(mock_app)
        with (
            patch("voicepad.tui.handlers.recording_handler.time.strftime", return_value="20220101_120000"),
            patch.object(Path, "write_text"),
            patch.object(Path, "mkdir"),
        ):
            handler.save_recording(audio, final_result)

        assert mock_app._current_text == "Complete ending at the moment."
        assert mock_button.disabled is False
        mock_format_markdown.assert_called_once()
        mock_format_streaming.assert_not_called()
        assert mock_app._entries[0].text == "Complete ending at the moment."
        assert mock_app._entries[0].device == "cpu"
        assert mock_app._entries[0].latency_ms == 321.0

    def test_adds_history_entry(self):
        """Test that save_recording adds history entry."""
        mock_app = Mock()
        chunk = ChunkResult(text="Test", is_final=True, device="cuda", latency_ms=100.0, segments=[], index=0)
        mock_app._stream_chunks = [chunk]
        mock_app._entries = []
        mock_app.config.recordings_path = Path("/tmp/recordings")
        mock_app.config.markdown_path = Path("/tmp/markdown")
        mock_app.config.recording_prefix = "recording"

        mock_button = Mock()
        mock_app.query_one.return_value = mock_button

        wav_path = Path("/tmp/recordings/recording_20220101_120000.wav")
        audio = _artifact(wav_path, 16_000)

        handler = RecordingHandler(mock_app)
        with (
            patch("voicepad.tui.handlers.recording_handler._format_markdown_streaming"),
            patch("voicepad.tui.handlers.recording_handler.time.strftime", return_value="20220101_120000"),
            patch.object(Path, "write_text"),
            patch.object(Path, "mkdir"),
        ):
            handler.save_recording(audio)

        assert len(mock_app._entries) == 1
        entry = mock_app._entries[0]
        assert entry.text == "Test"
        assert entry.device == "cuda"
        mock_app._add_history_entry.assert_called_once_with(entry)
