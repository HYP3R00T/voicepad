"""Tests for the VoicePad Textual application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from textual.widgets import Button, Label
from voicepad.tui.app import VoicePadApp, _format_markdown, _format_markdown_streaming
from voicepad.tui.workers import ModelWarmResult
from voicepad_core import ChunkResult, Config, Segment


@dataclass
class _FakeRecordingSession:
    """Minimal recording session stub for Textual interaction tests."""

    config: Config

    def start(self) -> None:
        """Pretend the microphone opened successfully."""

    def stop(self) -> np.ndarray:
        """Return a short buffer if the app ever reaches stop logic."""
        return np.zeros(16000, dtype=np.float32)


class TestVoicePadApp:
    async def test_mount_inserts_session_panels(self, monkeypatch, tmp_path: Path) -> None:
        """When the app mounts, the transcription panel contains its placeholder text."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#tx-text", Label).content == "speak and press space to begin…"

    async def test_mount_initializes_history_placeholder(self, monkeypatch, tmp_path: Path) -> None:
        """When the app mounts, the history panel starts with the empty-state hint."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#empty-hint", Label).content == "no recordings yet this session"

    async def test_on_model_ready_enables_recording(self, monkeypatch, tmp_path: Path) -> None:
        """When model warm-up succeeds, the record button becomes available."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cuda", compute_type="int8", fallback=False, error=None))
            assert app.query_one("#record-btn", Button).disabled is False

    async def test_space_starts_recording_when_model_is_ready(self, monkeypatch, tmp_path: Path) -> None:
        """When the user presses space and the model is ready, recording starts."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        monkeypatch.setattr("voicepad.tui.app.RecordingSession", _FakeRecordingSession)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cuda", compute_type="int8", fallback=False, error=None))
            await pilot.press("space")
            await pilot.pause()
            assert app.query_one("#status", Label).content == "◉  recording…"

    def test_format_markdown_with_segments_includes_segment_list(self, tmp_path: Path) -> None:
        """When transcription has segments, the markdown output includes a segment section."""
        wav_path = tmp_path / "clip.wav"
        result = SimpleNamespace(
            device="cuda",
            compute_type="int8",
            language="en",
            language_probability=0.99,
            duration_s=1.5,
            latency_ms=42.2,
            text="hello world",
            segments=[
                SimpleNamespace(start=0.0, end=0.8, text="hello"),
                SimpleNamespace(start=0.8, end=1.5, text="world"),
            ],
        )

        assert "## Segments" in _format_markdown(wav_path, result)

    def test_format_markdown_without_segments_omits_segment_list(self, tmp_path: Path) -> None:
        """When transcription has no segments, the markdown output skips the segment section."""
        wav_path = tmp_path / "clip.wav"
        result = SimpleNamespace(
            device="cpu",
            compute_type="int8",
            language="auto",
            language_probability=0.5,
            duration_s=0.7,
            latency_ms=12.0,
            text="",
            segments=[],
        )

        assert "## Segments" not in _format_markdown(wav_path, result)

    def test_format_markdown_streaming_includes_chunk_structure(self, tmp_path: Path) -> None:
        """When streaming chunks are present, the markdown keeps the same section layout."""
        wav_path = tmp_path / "clip.wav"
        chunks = [
            ChunkResult(
                index=1,
                text="hello world",
                segments=[Segment(start=0.0, end=1.0, text="hello world")],
                start_s=0.0,
                end_s=1.0,
                latency_ms=12.0,
                device="cuda",
                language="en",
                language_probability=0.98,
                is_final=True,
            )
        ]

        markdown = _format_markdown_streaming(wav_path, "hello world", 1.0, chunks)

        assert "**Mode:** streaming" in markdown
        assert "**Model:** cuda / live" in markdown
        assert "## Segments" in markdown
        assert "hello world" in markdown


class TestVoicePadAppExtended:
    async def test_on_model_ready_with_error_shows_error_status(self, monkeypatch, tmp_path: Path) -> None:
        """When model warm-up fails, the status shows the error message."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error="load failed"))
            await pilot.pause()
            status = app.query_one("#status", Label).content
            assert "error" in status or "load failed" in status

    async def test_on_model_ready_with_fallback_shows_cpu_fallback(self, monkeypatch, tmp_path: Path) -> None:
        """When model falls back to CPU, the header label reflects that."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=None))
            await pilot.pause()
            header_label = str(app.query_one("#header-model", Label).content)
            assert "cpu" in header_label.lower() or "fallback" in header_label.lower()

    async def test_toggle_recording_does_nothing_when_model_not_ready(self, monkeypatch, tmp_path: Path) -> None:
        """When the model is not ready, pressing space does not start recording."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            # Model not ready — space should be a no-op
            await pilot.press("space")
            await pilot.pause()
            # Status should still be initialising, not recording
            status = app.query_one("#status", Label).content
            assert "recording" not in status

    async def test_stop_recording_via_second_space(self, monkeypatch, tmp_path: Path) -> None:
        """Pressing space a second time while recording stops the recording."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        monkeypatch.setattr("voicepad.tui.app.RecordingSession", _FakeRecordingSession)

        # Stub out the transcription worker so it doesn't block.
        # _transcribe_worker is decorated with @work(thread=True) so it must be
        # a plain (non-async) callable when monkeypatched.
        monkeypatch.setattr(VoicePadApp, "_transcribe_worker", lambda self, audio: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False, error=None))
            await pilot.press("space")  # start
            await pilot.pause()
            await pilot.press("space")  # stop
            await pilot.pause()
            # After stopping, button should be in transcribing state (disabled)
            assert app.query_one("#record-btn", Button).disabled is True

    async def test_on_transcription_done_updates_tx_panel(self, monkeypatch, tmp_path: Path) -> None:
        """When transcription completes, the transcription panel is updated."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        result = SimpleNamespace(
            text="hello world",
            device="cpu",
            compute_type="int8",
            duration_s=1.0,
            latency_ms=50.0,
            language="en",
            language_probability=0.99,
            segments=[],
            fallback_to_cpu=False,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False, error=None))
            await pilot.pause()
            audio = np.zeros(16000, dtype=np.float32)
            app._on_transcription_done(audio, result, None)
            await pilot.pause()
            assert app.query_one("#tx-text", Label).content == "hello world"

    async def test_on_transcription_done_with_error_shows_error(self, monkeypatch, tmp_path: Path) -> None:
        """When transcription fails, the error is shown in the status."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False, error=None))
            await pilot.pause()
            audio = np.zeros(16000, dtype=np.float32)
            app._on_transcription_done(audio, None, "model failed")
            await pilot.pause()
            status = app.query_one("#status", Label).content
            assert "model failed" in status

    async def test_on_transcription_done_adds_history_entry(self, monkeypatch, tmp_path: Path) -> None:
        """When transcription succeeds, a history entry is added."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)

        result = SimpleNamespace(
            text="test entry",
            device="cpu",
            compute_type="int8",
            duration_s=1.0,
            latency_ms=50.0,
            language="en",
            language_probability=0.99,
            segments=[],
            fallback_to_cpu=False,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False, error=None))
            await pilot.pause()
            audio = np.zeros(16000, dtype=np.float32)
            app._on_transcription_done(audio, result, None)
            await pilot.pause()
            assert len(app._entries) == 1
            assert app._entries[0].text == "test entry"

    async def test_record_button_press_toggles_recording(self, monkeypatch, tmp_path: Path) -> None:
        """Clicking the record button starts recording when model is ready."""
        config = Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        monkeypatch.setattr("voicepad.tui.app.RecordingSession", _FakeRecordingSession)
        app = VoicePadApp(config)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False, error=None))
            await pilot.click("#record-btn")
            await pilot.pause()
            assert app.query_one("#status", Label).content == "◉  recording…"
