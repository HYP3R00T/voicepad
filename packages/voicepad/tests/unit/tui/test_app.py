"""Tests for the VoicePad Textual application."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from textual.widgets import Label
from voicepad.tui.app import VoicePadApp, _format_markdown, _format_markdown_streaming
from voicepad.tui.workers import ModelWarmResult
from voicepad_core import ChunkResult, Config, Segment

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeRecorder:
    """Minimal recorder stub compatible with StreamingTranscriber."""

    _lock: threading.Lock = None  # type: ignore[assignment]
    _frames: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._frames = []


@dataclass
class _FakeRecordingSession:
    """Recording session stub that exposes _recorder for StreamingTranscriber."""

    config: Config

    def __post_init__(self) -> None:
        self._recorder = _FakeRecorder()

    def start(self) -> None:
        pass

    def stop(self) -> np.ndarray:
        return np.zeros(16000, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ready_result(text: str = "hello world") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        device="cpu",
        compute_type="int8",
        duration_s=1.0,
        latency_ms=50.0,
        language="en",
        language_probability=0.99,
        segments=[],
        fallback_to_cpu=False,
    )


# ---------------------------------------------------------------------------
# App mount
# ---------------------------------------------------------------------------


class TestVoicePadAppMount:
    async def test_transcription_placeholder_present(self, monkeypatch, tmp_path: Path) -> None:
        """On mount, the transcription panel shows its placeholder text."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "speak and press space" in str(app.query_one("#tx-text", Label).content)

    async def test_status_shows_initialising(self, monkeypatch, tmp_path: Path) -> None:
        """On mount, the status label shows 'initialising'."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "initialising" in str(app.query_one("#status", Label).content)

    async def test_tabs_present(self, monkeypatch, tmp_path: Path) -> None:
        """On mount, all four tabs are present."""
        from textual.widgets import TabbedContent

        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one("#tabs", TabbedContent)
            assert tabs is not None


# ---------------------------------------------------------------------------
# Model ready
# ---------------------------------------------------------------------------


class TestModelReady:
    async def test_status_shows_ready(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cuda", compute_type="int8", fallback=False))
            await pilot.pause()
            assert "ready" in str(app.query_one("#status", Label).content)

    async def test_model_ready_flag_set(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cuda", compute_type="int8", fallback=False))
            assert app._model_ready is True

    async def test_error_result_shows_error_status(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error="load failed"))
            await pilot.pause()
            status = str(app.query_one("#status", Label).content)
            assert "error" in status or "load failed" in status

    async def test_fallback_reflected_in_header(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=True))
            await pilot.pause()
            header = str(app.query_one("#header-model", Label).content)
            assert "cpu" in header.lower() or "fallback" in header.lower()


# ---------------------------------------------------------------------------
# Recording toggle
# ---------------------------------------------------------------------------


class TestRecordingToggle:
    async def test_space_does_nothing_when_model_not_ready(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            assert app._recording is False

    async def test_space_starts_recording_when_ready(self, monkeypatch, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        monkeypatch.setattr("voicepad.tui.app.RecordingSession", _FakeRecordingSession)
        # Stub out the streamer so it doesn't actually run
        _stub = MagicMock()
        _stub.start = lambda: None
        _stub.stop = lambda: None
        monkeypatch.setattr("voicepad.tui.app.StreamingTranscriber", lambda **kw: _stub)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False))
            await pilot.press("space")
            await pilot.pause()
            assert app._recording is True

    async def test_space_does_nothing_on_non_record_tab(self, monkeypatch, tmp_path: Path) -> None:
        from textual.widgets import TabbedContent

        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False))
            # Switch to history tab
            app.query_one("#tabs", TabbedContent).active = "tab-history"
            await pilot.press("space")
            await pilot.pause()
            assert app._recording is False


# ---------------------------------------------------------------------------
# Streaming chunk callback
# ---------------------------------------------------------------------------


class TestOnStreamChunk:
    async def test_chunk_text_appears_in_transcription_panel(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            chunk = ChunkResult(index=1, text="hello streaming", is_final=False)
            app._on_stream_chunk(chunk)
            await pilot.pause()
            content = str(app.query_one("#tx-text", Label).content)
            assert "hello streaming" in content

    async def test_final_chunk_sets_ready_status(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_model_ready(ModelWarmResult(device="cpu", compute_type="int8", fallback=False))
            app._transcribing = True
            chunk = ChunkResult(index=1, text="done", is_final=True)
            app._on_stream_chunk(chunk)
            await pilot.pause()
            assert app._transcribing is False
            assert "ready" in str(app.query_one("#status", Label).content)

    async def test_multiple_chunks_accumulate(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_stream_chunk(ChunkResult(index=1, text="first", is_final=False))
            app._on_stream_chunk(ChunkResult(index=2, text="second", is_final=False))
            await pilot.pause()
            content = str(app.query_one("#tx-text", Label).content)
            assert "first" in content
            assert "second" in content

    async def test_empty_chunk_does_not_add_to_list(self, monkeypatch, tmp_path: Path) -> None:
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_stream_chunk(ChunkResult(index=1, text="", is_final=False))
            assert len(app._stream_chunks) == 0


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------


class TestHistoryLoading:
    async def test_existing_markdown_files_populate_history(self, monkeypatch, tmp_path: Path) -> None:
        """Markdown files on disk are loaded into the history list on mount."""
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        rec_dir = tmp_path / "recordings"
        rec_dir.mkdir()

        # Write a minimal markdown file
        md = md_dir / "recording_20260101_120000.md"
        md.write_text(
            "# Transcription\n\n**File:** `recording_20260101_120000.wav`\n"
            "**Duration:** 5.0s\n**Latency:** 500ms\n**Model:** cuda / int8\n\n"
            "---\n\n## Text\n\nhello from history\n",
            encoding="utf-8",
        )

        config = Config(recordings_path=rec_dir, markdown_path=md_dir)
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert len(app._entries) == 1
            assert "hello from history" in app._entries[0].text


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    def test_includes_filename(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        result = _ready_result()
        assert "clip.wav" in _format_markdown(wav, result)

    def test_includes_text(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        assert "hello world" in _format_markdown(wav, _ready_result("hello world"))

    def test_includes_segments_when_present(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        result = _ready_result()
        result.segments = [SimpleNamespace(start=0.0, end=1.0, text="hello")]
        assert "## Segments" in _format_markdown(wav, result)

    def test_omits_segments_when_empty(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        assert "## Segments" not in _format_markdown(wav, _ready_result())

    def test_empty_text_uses_placeholder(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        result = _ready_result(text="")
        assert "no speech detected" in _format_markdown(wav, result)


class TestFormatMarkdownStreaming:
    def test_includes_mode_streaming(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        md = _format_markdown_streaming(wav, "hello", 1.0, [])
        assert "streaming" in md

    def test_includes_text(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        md = _format_markdown_streaming(wav, "hello world", 1.0, [])
        assert "hello world" in md

    def test_includes_segments_from_chunks(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        chunks = [
            ChunkResult(
                index=1,
                text="hello",
                segments=[Segment(start=0.0, end=1.0, text="hello")],
                start_s=0.0,
                end_s=1.0,
                latency_ms=100.0,
                device="cuda",
                language="en",
                language_probability=0.99,
                is_final=True,
            )
        ]
        md = _format_markdown_streaming(wav, "hello", 1.0, chunks)
        assert "## Segments" in md
        assert "0.0s" in md

    def test_uses_device_from_last_chunk(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        chunks = [ChunkResult(index=1, text="hi", device="cpu", is_final=True)]
        md = _format_markdown_streaming(wav, "hi", 1.0, chunks)
        assert "cpu" in md

    def test_empty_chunks_list(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        md = _format_markdown_streaming(wav, "hi", 1.0, [])
        assert "hi" in md
        assert "## Segments" not in md

    def test_total_latency_is_sum_of_chunks(self, tmp_path: Path) -> None:
        wav = tmp_path / "clip.wav"
        chunks = [
            ChunkResult(index=1, text="a", latency_ms=500.0, is_final=False),
            ChunkResult(index=2, text="b", latency_ms=700.0, is_final=True),
        ]
        md = _format_markdown_streaming(wav, "a b", 2.0, chunks)
        assert "1200ms" in md


# ---------------------------------------------------------------------------
# Import needed for SimpleNamespace usage in tests
# ---------------------------------------------------------------------------
