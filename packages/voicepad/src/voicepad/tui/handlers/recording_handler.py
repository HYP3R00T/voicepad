"""Recording handler for VoicePad TUI."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from textual.widgets import Label, Static, TabbedContent
from voicepad_core import (
    ChunkResult,
    MicrophoneStream,
    StreamingTranscriber,
    begin_transcription_session,
    end_transcription_session,
)

from voicepad.tui.components import VoiceButton
from voicepad.tui.utils.clipboard import copy_to_clipboard as _copy_to_clipboard
from voicepad.tui.utils.markdown import format_markdown as _format_markdown
from voicepad.tui.utils.markdown import format_markdown_streaming as _format_markdown_streaming
from voicepad.tui.workers import RecordingSession

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp


class RecordingHandler:
    """Handles recording and transcription functionality."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app
        self._final_chunk_event: threading.Event | None = None
        self._session_logger = None
        self._log_file = None

    def action_toggle_recording(self) -> None:
        """Toggle recording on/off (space bar binding)."""
        active = self.app.query_one("#tabs", TabbedContent).active
        if active != "tab-record":
            return
        if not self.app._model_ready or self.app._transcribing:
            return
        if self.app._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        """Start recording audio."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_level = getattr(self.app.config, "log_level", "INFO")
        if not isinstance(log_level, str):
            log_level = "INFO"

        self._session_logger, self._log_file = begin_transcription_session(
            logs_path=self.app.config.logs_path,
            log_level=log_level,
            session_id=f"streaming_{session_id}",
            include_streaming=True,
        )

        self._session_logger.info("Starting recording session")

        self._final_chunk_event = threading.Event()
        self.app._session = RecordingSession(config=self.app.config)
        try:
            self.app._session.start()
            self._session_logger.info("Recording session started successfully")
        except Exception as e:
            self._session_logger.error(f"Failed to start recording: {e}")
            self.app._set_status("error", f"mic error: {e}")
            return

        self.app._recording = True
        self.app._record_start = time.monotonic()
        self.app._stream_chunks = []
        self.app._set_status("recording", "recording…")
        self.app._start_timer()

        # Start streaming transcriber — transcribes chunks during recording
        recorder = self.app._session._recorder
        if recorder is None:
            self._session_logger.error("Recorder is None, cannot start streaming")
            return

        self._session_logger.info(
            f"Starting streaming transcriber: model={self.app.config.transcription_model}, "
            f"device={self.app.config.transcription_device}, "
            f"min_chunk={self.app.config.min_chunk_s}s, per_chunk_max={self.app.config.max_chunk_s}s"
        )

        self.app._streamer = StreamingTranscriber(
            recorder=recorder,
            on_chunk=lambda chunk: self.app.call_from_thread(self._handle_stream_chunk, chunk),
            on_error=lambda err: self.app.call_from_thread(self._handle_stream_error, err),
            model_name=self.app.config.transcription_model,
            device=self.app.config.transcription_device,
            compute_type=self.app.config.transcription_compute_type,
            min_chunk_s=self.app.config.min_chunk_s,
            max_chunk_s=self.app.config.max_chunk_s,
            overlap_s=self.app.config.overlap_s,
            silence_threshold_ms=self.app.config.silence_threshold_ms,
        )
        self.app._streamer.start()
        self._session_logger.info("Streaming transcriber started")

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if self.app._session is None:
            return

        if self._session_logger:
            self._session_logger.info("Stopping recording session")

        self.app._recording = False
        self.app._stop_timer()
        self.app._set_status("transcribing", "transcribing…")

        try:
            audio = self.app._session.stop()
            if self._session_logger:
                self._session_logger.info(f"Recording stopped, audio length: {len(audio)} samples")
        except Exception as e:
            if self._session_logger:
                self._session_logger.error(f"Failed to stop recording: {e}")
            self.app._set_status("error", f"stop error: {e}")
            if self.app._streamer:
                self.app._streamer._stop_event.set()
            return

        self.app._transcribing = True
        # Stop the streamer in a thread — it will transcribe the tail and call on_chunk(is_final=True)
        # Must go through the app's @work-decorated wrapper so it runs in a background thread.
        self.app._finalize_worker(audio)

    def finalize_worker(self, audio: np.ndarray) -> None:
        """Stop the streamer (transcribes tail) then save the full recording."""
        if self._session_logger:
            self._session_logger.info("Finalizing transcription...")

        if self.app._streamer:
            self.app._streamer.stop()  # blocks until final chunk callback fires
            if self._session_logger:
                self._session_logger.info("Streamer stopped")

        if self._final_chunk_event is not None:
            self._final_chunk_event.wait(timeout=5.0)

        final_result = self._transcribe_final_audio(audio)
        self.app.call_from_thread(self.save_recording, audio, final_result)

        # Clear the session logger
        end_transcription_session(include_streaming=True)

        if self._session_logger and self._log_file:
            self._session_logger.info("=" * 80)
            self._session_logger.info(f"Session complete. Log saved to: {self._log_file}")
            self._session_logger.info("=" * 80)

    def _transcribe_final_audio(self, audio: np.ndarray):
        """Use the fully recorded audio as the final authority for short/simple sessions."""
        if len(self.app._stream_chunks) > 1:
            return None

        from voicepad_core import transcribe

        try:
            if self._session_logger:
                self._session_logger.info("Running final full-audio transcription pass")
            return transcribe(
                audio,
                model_name=self.app.config.transcription_model,
                device=self.app.config.transcription_device,
                compute_type=self.app.config.transcription_compute_type,
                language=self.app.config.language,
                word_timestamps=False,
            )
        except Exception as e:
            if self._session_logger:
                self._session_logger.warning(f"Final full-audio pass failed, keeping streaming result: {e}")
            return None

    def _handle_stream_chunk(self, chunk: ChunkResult) -> None:
        """Handle streamed chunks on the main thread and signal completion for the final chunk."""
        self.on_stream_chunk(chunk)
        if chunk.is_final and self._final_chunk_event is not None:
            self._final_chunk_event.set()

    def _handle_stream_error(self, error: str) -> None:
        """Handle a chunk transcription error and release the recording state."""
        self.app._transcribing = False
        self.app._set_status("error", error)
        if self._final_chunk_event is not None:
            self._final_chunk_event.set()

    def on_stream_chunk(self, chunk: ChunkResult) -> None:
        """Called from the streaming thread for each transcribed chunk."""
        if chunk.text:
            self.app._stream_chunks.append(chunk)

        # Update transcription panel with all accumulated text so far
        full_text = " ".join(c.text for c in self.app._stream_chunks).strip()
        if full_text:
            tx_text = self.app.query_one("#tx-text", Label)
            tx_text.remove_class("placeholder")
            tx_text.update(full_text)
            self.app.query_one("#transcription", Static).scroll_end(animate=False)

        if chunk.is_final:
            self.app._transcribing = False
            elapsed = time.monotonic() - self.app._record_start
            self.app.query_one("#tx-meta", Label).update(f"[dim]{elapsed:.1f}s  ·  streaming[/]")
            self.app._set_status("ready", "ready")
            # Auto-copy if triggered by global hotkey
            if self.app._hotkey_pending_copy:
                self.app._hotkey_pending_copy = False
                full_text = " ".join(c.text for c in self.app._stream_chunks).strip()
                if full_text:
                    _copy_to_clipboard(full_text)
                    self.app._set_status("ready", "ready — copied to clipboard")
                    self.app.set_timer(2.0, lambda: self.app._set_status("ready", "ready"))
                    self.app._overlay_set("copied")
                else:
                    self.app._overlay_set("hidden")
            else:
                self.app._overlay_set("hidden")

    def save_recording(self, audio: np.ndarray, final_result: object | None = None) -> None:
        """Save WAV + markdown and add history entry after streaming completes."""
        from voicepad.tui.models import SessionEntry

        final_text = getattr(final_result, "text", "") if final_result is not None else ""
        full_text = final_text or " ".join(c.text for c in self.app._stream_chunks).strip()
        self.app._current_text = full_text
        copy_button = self.app.query_one("#tx-copy-btn", VoiceButton)
        copy_button.disabled = not bool(full_text)

        # Save WAV
        wav_path: Path | None = None
        md_path: Path | None = None
        recorder_ref: MicrophoneStream | None = self.app._session._recorder if self.app._session else None
        if recorder_ref is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            wav_path = self.app.config.recordings_path / f"{self.app.config.recording_prefix}_{ts}.wav"
            try:
                recorder_ref.save_wav(audio, wav_path, sample_rate=16000)
                if full_text:
                    md_path = self.app.config.markdown_path / f"{wav_path.stem}.md"
                    self.app.config.markdown_path.mkdir(parents=True, exist_ok=True)
                    duration_s = len(audio) / 16000
                    markdown = (
                        _format_markdown(wav_path, final_result, self.app.config.transcription_model)
                        if final_result is not None
                        else _format_markdown_streaming(
                            wav_path,
                            full_text,
                            duration_s,
                            self.app._stream_chunks,
                            self.app.config.transcription_model,
                        )
                    )
                    md_path.write_text(markdown, encoding="utf-8")
            except Exception:
                wav_path = None
                md_path = None

        entry = SessionEntry(
            index=len(self.app._entries),
            wav_path=wav_path,
            md_path=md_path,
            duration_s=len(audio) / 16000,
            text=full_text,
            latency_ms=float(getattr(final_result, "latency_ms", 0.0)),
            device=str(
                getattr(
                    final_result, "device", self.app._stream_chunks[-1].device if self.app._stream_chunks else "cuda"
                )
            ),
        )
        self.app._entries.append(entry)
        self.app._add_history_entry(entry)
