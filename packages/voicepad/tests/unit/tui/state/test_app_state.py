"""Tests for AppState."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from voicepad.tui.models import SessionEntry
from voicepad.tui.state.app_state import AppState
from voicepad.tui.workers import ModelWarmResult, RecordingSession


class TestAppStateInitialization:
    """Test suite for AppState initialization."""

    def test_init_creates_default_state(self) -> None:
        """AppState initializes with default values."""
        state = AppState()

        # Model state
        assert state.model_ready is False
        assert state.warm_result is None

        # Recording state
        assert state.recording is False
        assert state.transcribing is False
        assert state.session is None
        assert state.record_start == 0.0

        # Streaming state
        assert state.streamer is None
        assert state.stream_chunks == []

        # Transcription state
        assert state.current_text == ""

        # History state
        assert state.entries == []
        assert state.selected_entry_idx is None

        # Hotkey state
        assert state.hotkey_listener is None
        assert state.hotkey_pending_copy is False
        assert state.overlay is None

    def test_init_with_custom_values(self) -> None:
        """AppState can be initialized with custom values."""
        warm_result = ModelWarmResult(device="cuda", compute_type="float16", fallback=False)
        entry = SessionEntry(
            index=0,
            wav_path=Path("/test.wav"),
            md_path=Path("/test.md"),
            duration_s=5.0,
            text="test",
            latency_ms=100.0,
            device="cuda",
        )

        state = AppState(
            model_ready=True,
            warm_result=warm_result,
            recording=True,
            transcribing=True,
            record_start=123.45,
            current_text="hello",
            entries=[entry],
            selected_entry_idx=0,
            hotkey_pending_copy=True,
        )

        assert state.model_ready is True
        assert state.warm_result == warm_result
        assert state.recording is True
        assert state.transcribing is True
        assert state.record_start == 123.45
        assert state.current_text == "hello"
        assert len(state.entries) == 1
        assert state.entries[0] == entry
        assert state.selected_entry_idx == 0
        assert state.hotkey_pending_copy is True

    def test_init_creates_independent_lists(self) -> None:
        """Each AppState instance has independent list instances."""
        state1 = AppState()
        state2 = AppState()

        # Modify state1's lists
        state1.stream_chunks.append(MagicMock())
        state1.entries.append(
            SessionEntry(
                index=0,
                wav_path=None,
                md_path=None,
                duration_s=0.0,
                text="",
                latency_ms=0.0,
                device="cpu",
            )
        )

        # state2's lists should be unaffected
        assert len(state2.stream_chunks) == 0
        assert len(state2.entries) == 0


class TestAppStateModelState:
    """Test suite for model-related state."""

    def test_model_ready_flag(self) -> None:
        """model_ready flag can be set and read."""
        state = AppState()
        assert state.model_ready is False

        state.model_ready = True
        assert state.model_ready is True

    def test_warm_result_storage(self) -> None:
        """warm_result stores ModelWarmResult."""
        state = AppState()
        result = ModelWarmResult(device="cuda", compute_type="float16", fallback=False)

        state.warm_result = result
        assert state.warm_result == result
        assert state.warm_result.device == "cuda"
        assert state.warm_result.compute_type == "float16"
        assert state.warm_result.fallback is False

    def test_warm_result_with_error(self) -> None:
        """warm_result can store error information."""
        state = AppState()
        result = ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error="GPU not available")

        state.warm_result = result
        assert state.warm_result.error == "GPU not available"
        assert state.warm_result.fallback is True


class TestAppStateRecordingState:
    """Test suite for recording-related state."""

    def test_recording_flag(self) -> None:
        """recording flag can be set and read."""
        state = AppState()
        assert state.recording is False

        state.recording = True
        assert state.recording is True

    def test_transcribing_flag(self) -> None:
        """transcribing flag can be set and read."""
        state = AppState()
        assert state.transcribing is False

        state.transcribing = True
        assert state.transcribing is True

    def test_session_storage(self) -> None:
        """session stores RecordingSession."""
        state = AppState()
        config = MagicMock()
        session = RecordingSession(config=config)

        state.session = session
        assert state.session == session
        assert state.session.config == config

    def test_record_start_timestamp(self) -> None:
        """record_start stores timestamp."""
        state = AppState()
        timestamp = 1234567890.123

        state.record_start = timestamp
        assert state.record_start == timestamp


class TestAppStateStreamingState:
    """Test suite for streaming-related state."""

    def test_streamer_storage(self) -> None:
        """streamer stores StreamingTranscriber."""
        state = AppState()
        streamer = MagicMock()

        state.streamer = streamer
        assert state.streamer == streamer

    def test_stream_chunks_list(self) -> None:
        """stream_chunks stores list of ChunkResult."""
        state = AppState()
        chunk1 = MagicMock()
        chunk2 = MagicMock()

        state.stream_chunks.append(chunk1)
        state.stream_chunks.append(chunk2)

        assert len(state.stream_chunks) == 2
        assert state.stream_chunks[0] == chunk1
        assert state.stream_chunks[1] == chunk2

    def test_stream_chunks_can_be_cleared(self) -> None:
        """stream_chunks list can be cleared."""
        state = AppState()
        state.stream_chunks.append(MagicMock())
        state.stream_chunks.append(MagicMock())

        state.stream_chunks.clear()
        assert len(state.stream_chunks) == 0


class TestAppStateTranscriptionState:
    """Test suite for transcription-related state."""

    def test_current_text_storage(self) -> None:
        """current_text stores transcription text."""
        state = AppState()
        text = "This is a test transcription"

        state.current_text = text
        assert state.current_text == text

    def test_current_text_can_be_updated(self) -> None:
        """current_text can be updated multiple times."""
        state = AppState()

        state.current_text = "First"
        assert state.current_text == "First"

        state.current_text = "Second"
        assert state.current_text == "Second"

    def test_current_text_can_be_empty(self) -> None:
        """current_text can be empty string."""
        state = AppState()
        state.current_text = "Some text"

        state.current_text = ""
        assert state.current_text == ""


class TestAppStateHistoryState:
    """Test suite for history-related state."""

    def test_entries_list(self) -> None:
        """entries stores list of SessionEntry."""
        state = AppState()
        entry1 = SessionEntry(
            index=0,
            wav_path=Path("/test1.wav"),
            md_path=Path("/test1.md"),
            duration_s=5.0,
            text="first",
            latency_ms=100.0,
            device="cuda",
        )
        entry2 = SessionEntry(
            index=1,
            wav_path=Path("/test2.wav"),
            md_path=Path("/test2.md"),
            duration_s=3.0,
            text="second",
            latency_ms=150.0,
            device="cuda",
        )

        state.entries.append(entry1)
        state.entries.append(entry2)

        assert len(state.entries) == 2
        assert state.entries[0] == entry1
        assert state.entries[1] == entry2

    def test_selected_entry_idx_storage(self) -> None:
        """selected_entry_idx stores selected index."""
        state = AppState()

        state.selected_entry_idx = 5
        assert state.selected_entry_idx == 5

    def test_selected_entry_idx_can_be_none(self) -> None:
        """selected_entry_idx can be None."""
        state = AppState()
        state.selected_entry_idx = 3

        state.selected_entry_idx = None
        assert state.selected_entry_idx is None

    def test_entries_can_be_cleared(self) -> None:
        """entries list can be cleared."""
        state = AppState()
        state.entries.append(
            SessionEntry(
                index=0,
                wav_path=None,
                md_path=None,
                duration_s=0.0,
                text="",
                latency_ms=0.0,
                device="cpu",
            )
        )

        state.entries.clear()
        assert len(state.entries) == 0


class TestAppStateHotkeyState:
    """Test suite for hotkey-related state."""

    def test_hotkey_listener_storage(self) -> None:
        """hotkey_listener stores listener object."""
        state = AppState()
        listener = MagicMock()

        state.hotkey_listener = listener
        assert state.hotkey_listener == listener

    def test_hotkey_pending_copy_flag(self) -> None:
        """hotkey_pending_copy flag can be set and read."""
        state = AppState()
        assert state.hotkey_pending_copy is False

        state.hotkey_pending_copy = True
        assert state.hotkey_pending_copy is True

    def test_overlay_storage(self) -> None:
        """overlay stores overlay object."""
        state = AppState()
        overlay = MagicMock()

        state.overlay = overlay
        assert state.overlay == overlay


class TestAppStateResetRecordingState:
    """Test suite for reset_recording_state method."""

    def test_reset_recording_state_clears_flags(self) -> None:
        """reset_recording_state clears recording and transcribing flags."""
        state = AppState()
        state.recording = True
        state.transcribing = True

        state.reset_recording_state()

        assert state.recording is False
        assert state.transcribing is False

    def test_reset_recording_state_clears_session(self) -> None:
        """reset_recording_state clears session."""
        state = AppState()
        config = MagicMock()
        state.session = RecordingSession(config=config)

        state.reset_recording_state()

        assert state.session is None

    def test_reset_recording_state_clears_record_start(self) -> None:
        """reset_recording_state resets record_start to 0.0."""
        state = AppState()
        state.record_start = 1234567890.123

        state.reset_recording_state()

        assert state.record_start == 0.0

    def test_reset_recording_state_clears_stream_chunks(self) -> None:
        """reset_recording_state clears stream_chunks list."""
        state = AppState()
        state.stream_chunks.append(MagicMock())
        state.stream_chunks.append(MagicMock())

        state.reset_recording_state()

        assert len(state.stream_chunks) == 0

    def test_reset_recording_state_preserves_other_state(self) -> None:
        """reset_recording_state preserves non-recording state."""
        state = AppState()
        state.model_ready = True
        state.current_text = "preserved"
        state.selected_entry_idx = 5
        state.hotkey_pending_copy = True

        state.reset_recording_state()

        assert state.model_ready is True
        assert state.current_text == "preserved"
        assert state.selected_entry_idx == 5
        assert state.hotkey_pending_copy is True

    def test_reset_recording_state_can_be_called_multiple_times(self) -> None:
        """reset_recording_state can be called multiple times safely."""
        state = AppState()
        state.recording = True
        state.transcribing = True
        state.record_start = 123.45

        state.reset_recording_state()
        state.reset_recording_state()  # Second call should not error

        assert state.recording is False
        assert state.transcribing is False
        assert state.record_start == 0.0


class TestAppStateResetStreamingState:
    """Test suite for reset_streaming_state method."""

    def test_reset_streaming_state_clears_streamer(self) -> None:
        """reset_streaming_state clears streamer."""
        state = AppState()
        state.streamer = MagicMock()

        state.reset_streaming_state()

        assert state.streamer is None

    def test_reset_streaming_state_clears_stream_chunks(self) -> None:
        """reset_streaming_state clears stream_chunks list."""
        state = AppState()
        state.stream_chunks.append(MagicMock())
        state.stream_chunks.append(MagicMock())

        state.reset_streaming_state()

        assert len(state.stream_chunks) == 0

    def test_reset_streaming_state_preserves_other_state(self) -> None:
        """reset_streaming_state preserves non-streaming state."""
        state = AppState()
        state.recording = True
        state.transcribing = True
        state.current_text = "preserved"
        state.model_ready = True

        state.reset_streaming_state()

        assert state.recording is True
        assert state.transcribing is True
        assert state.current_text == "preserved"
        assert state.model_ready is True

    def test_reset_streaming_state_can_be_called_multiple_times(self) -> None:
        """reset_streaming_state can be called multiple times safely."""
        state = AppState()
        state.streamer = MagicMock()
        state.stream_chunks.append(MagicMock())

        state.reset_streaming_state()
        state.reset_streaming_state()  # Second call should not error

        assert state.streamer is None
        assert len(state.stream_chunks) == 0


class TestAppStateIntegration:
    """Integration tests for AppState."""

    def test_complete_recording_workflow(self) -> None:
        """AppState supports complete recording workflow."""
        state = AppState()

        # Start recording
        config = MagicMock()
        state.session = RecordingSession(config=config)
        state.recording = True
        state.record_start = 1234567890.0

        assert state.recording is True
        assert state.session is not None

        # Start transcribing
        state.transcribing = True
        state.streamer = MagicMock()
        state.stream_chunks.append(MagicMock())

        assert state.transcribing is True
        assert len(state.stream_chunks) == 1

        # Complete recording
        state.reset_recording_state()

        assert state.recording is False
        assert state.transcribing is False
        assert state.session is None
        assert len(state.stream_chunks) == 0

    def test_model_warm_and_transcription_workflow(self) -> None:
        """AppState supports model warming and transcription workflow."""
        state = AppState()

        # Warm model
        state.warm_result = ModelWarmResult(device="cuda", compute_type="float16", fallback=False)
        state.model_ready = True

        assert state.model_ready is True
        assert state.warm_result.device == "cuda"

        # Transcribe
        state.current_text = "Hello world"

        # Add to history
        entry = SessionEntry(
            index=0,
            wav_path=Path("/test.wav"),
            md_path=Path("/test.md"),
            duration_s=2.5,
            text=state.current_text,
            latency_ms=100.0,
            device=state.warm_result.device,
        )
        state.entries.append(entry)
        state.selected_entry_idx = 0

        assert len(state.entries) == 1
        assert state.entries[0].text == "Hello world"
        assert state.selected_entry_idx == 0

    def test_hotkey_workflow(self) -> None:
        """AppState supports hotkey workflow."""
        state = AppState()

        # Set up hotkey
        state.hotkey_listener = MagicMock()
        state.overlay = MagicMock()

        # Trigger copy
        state.hotkey_pending_copy = True
        assert state.hotkey_pending_copy is True

        # Complete copy
        state.hotkey_pending_copy = False
        assert state.hotkey_pending_copy is False
