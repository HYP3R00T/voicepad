"""Centralized application state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voicepad_core import ChunkResult, StreamingTranscriber

from voicepad.tui.models import SessionEntry
from voicepad.tui.workers import ModelWarmResult, RecordingSession


@dataclass
class AppState:
    """Central state container for VoicePad app."""

    # Model state
    model_ready: bool = False
    warm_result: ModelWarmResult | None = None

    # Recording state
    recording: bool = False
    transcribing: bool = False
    session: RecordingSession | None = None
    record_start: float = 0.0

    # Streaming state
    streamer: StreamingTranscriber | None = None
    stream_chunks: list[ChunkResult] = field(default_factory=list)

    # Transcription state
    current_text: str = ""

    # History state
    entries: list[SessionEntry] = field(default_factory=list)
    selected_entry_idx: int | None = None

    # Hotkey state
    hotkey_listener: Any = None
    hotkey_pending_copy: bool = False
    overlay: Any = None

    def reset_recording_state(self) -> None:
        """Reset state after recording completes."""
        self.recording = False
        self.transcribing = False
        self.session = None
        self.record_start = 0.0
        self.stream_chunks.clear()

    def reset_streaming_state(self) -> None:
        """Reset streaming transcription state."""
        self.streamer = None
        self.stream_chunks.clear()
