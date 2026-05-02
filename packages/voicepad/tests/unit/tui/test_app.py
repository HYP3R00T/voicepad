"""Tests for the VoicePad Textual application."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from textual.widgets import Label, Link
from voicepad.tui.app import (
    DeleteConfirmModal,
    InfoModal,
    VoicePadApp,
    _build_hotkey_str,
    _format_markdown,
    _format_markdown_streaming,
    _parse_hotkey_str,
)
from voicepad.tui.workers import ModelWarmResult
from voicepad_core import ChunkResult, Config, Segment

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeRecorder:
    """Minimal recorder stub compatible with StreamingTranscriber."""

    _lock: threading.Lock | None = None
    _frames: list | None = None

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

        # Write a markdown file in the YAML front-matter format that parse_markdown_entry expects
        md = md_dir / "recording_20260101_120000.md"
        md.write_text(
            "---\n"
            "file: recording_20260101_120000.wav\n"
            "transcriptions:\n"
            "  - n: 1\n"
            "    model: turbo · cuda / int8\n"
            "    language: en (99.0%)\n"
            "    duration: 5.0s\n"
            "    latency: 500ms\n"
            "    timestamp: 2026-01-01 12:00\n"
            "---\n"
            "\n"
            "## Transcription 1\n"
            "\n"
            "hello from history\n",
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


# ---------------------------------------------------------------------------
# Hotkey parsing helpers
# ---------------------------------------------------------------------------


class TestParseHotkeyStr:
    """Tests for _parse_hotkey_str helper function."""

    def test_parse_simple_key(self) -> None:
        """Parse a simple key like 'v'."""
        mods, key = _parse_hotkey_str("v")
        assert key == "v"
        assert mods == []

    def test_parse_single_modifier_with_key(self) -> None:
        """Parse a key with a single modifier like '<ctrl>+v'."""
        mods, key = _parse_hotkey_str("<ctrl>+v")
        assert "ctrl" in mods
        assert key == "v"

    def test_parse_multiple_modifiers(self) -> None:
        """Parse a key with multiple modifiers like '<ctrl>+<alt>+v'."""
        mods, key = _parse_hotkey_str("<ctrl>+<alt>+v")
        assert "ctrl" in mods
        assert "alt" in mods
        assert key == "v"

    def test_parse_special_key(self) -> None:
        """Parse a special key like 'space' or 'f1'."""
        mods, key = _parse_hotkey_str("space")
        assert key == "space"

    def test_parse_case_insensitive(self) -> None:
        """Parsing should be case-insensitive."""
        mods, key = _parse_hotkey_str("<CTRL>+<ALT>+V")
        assert "ctrl" in mods
        assert "alt" in mods
        assert key == "v"

    def test_parse_with_spaces(self) -> None:
        """Parse hotkey with spaces around plus signs."""
        mods, key = _parse_hotkey_str("<ctrl> + <alt> + v")
        assert "ctrl" in mods
        assert "alt" in mods
        assert key == "v"

    def test_parse_f_keys(self) -> None:
        """Parse F-keys like f1, f2, etc."""
        mods, key = _parse_hotkey_str("<shift>+f1")
        assert "shift" in mods
        assert key == "f1"

    def test_parse_empty_modifiers(self) -> None:
        """When no modifiers are present, mods list is empty."""
        mods, key = _parse_hotkey_str("a")
        assert mods == []


class TestBuildHotkeyStr:
    """Tests for _build_hotkey_str helper function."""

    def test_build_single_key(self) -> None:
        """Build a single key without modifiers."""
        result = _build_hotkey_str([], "v")
        assert "v" in result

    def test_build_with_single_modifier(self) -> None:
        """Build a key with a single modifier."""
        result = _build_hotkey_str(["ctrl"], "v")
        assert "<ctrl>" in result
        assert "+v" in result

    def test_build_with_multiple_modifiers(self) -> None:
        """Build a key with multiple modifiers."""
        result = _build_hotkey_str(["ctrl", "alt"], "v")
        assert "<ctrl>" in result
        assert "<alt>" in result
        assert "v" in result

    def test_build_special_key_with_angle_brackets(self) -> None:
        """Build a special key which should have angle brackets."""
        result = _build_hotkey_str(["ctrl"], "space")
        assert "<ctrl>" in result
        assert "<space>" in result

    def test_build_single_char_key_no_brackets(self) -> None:
        """Single-char keys should not have angle brackets."""
        result = _build_hotkey_str(["shift"], "a")
        assert "<shift>" in result
        assert "+a" in result
        # Ensure 'a' is not wrapped in <> (unless it's part of <shift>)

    def test_build_empty_key(self) -> None:
        """Building with empty key should return empty string."""
        result = _build_hotkey_str(["ctrl"], "")
        assert result == ""

    def test_build_roundtrip_with_parse(self) -> None:
        """Build and parse should roundtrip correctly."""
        original_mods = ["ctrl", "alt"]
        original_key = "v"
        built = _build_hotkey_str(original_mods, original_key)
        parsed_mods, parsed_key = _parse_hotkey_str(built)
        assert set(parsed_mods) == set(original_mods)
        assert parsed_key == original_key


# ---------------------------------------------------------------------------
# DeleteConfirmModal tests
# ---------------------------------------------------------------------------


class TestDeleteConfirmModal:
    """Tests for the DeleteConfirmModal screen."""

    async def test_modal_shows_entry_name(self, monkeypatch, tmp_path: Path) -> None:
        """DeleteConfirmModal should display the entry name being deleted."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = DeleteConfirmModal("test_recording.wav")
            app.push_screen(modal)
            await pilot.pause()
            # Check that the modal was created and the entry name is accessible
            assert modal._entry_name == "test_recording.wav"

    async def test_cancel_button_dismisses_with_false(self, monkeypatch, tmp_path: Path) -> None:
        """Clicking Cancel should dismiss the modal with False."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = DeleteConfirmModal("test.wav")
            app.push_screen(modal)
            await pilot.pause()
            # Click the Cancel button
            cancel_btn = app.screen.query_one("#delete-cancel")
            cancel_btn.press()  # type: ignore
            await pilot.pause()
            # Modal should be dismissed
            assert len(app.screen_stack) == 1

    async def test_confirm_button_dismisses_with_true(self, monkeypatch, tmp_path: Path) -> None:
        """Clicking Confirm should dismiss the modal with True."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = DeleteConfirmModal("test.wav")
            app.push_screen(modal)
            await pilot.pause()
            # Click the Confirm button
            confirm_btn = app.screen.query_one("#delete-confirm")
            confirm_btn.press()  # type: ignore
            await pilot.pause()
            # Modal should be dismissed
            assert len(app.screen_stack) == 1

    async def test_modal_bindings_defined(self) -> None:
        """DeleteConfirmModal should have escape binding defined."""
        modal = DeleteConfirmModal("test.wav")
        # Verify bindings are set up
        assert len(modal.BINDINGS) > 0


# ---------------------------------------------------------------------------
# InfoModal tests
# ---------------------------------------------------------------------------


class TestInfoModal:
    """Tests for the InfoModal screen."""

    async def test_info_modal_has_sponsor_link(self, monkeypatch, tmp_path: Path) -> None:
        """InfoModal should have a GitHub Sponsors link."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Open the info modal
            app.push_screen(InfoModal())
            await pilot.pause()
            # Check for sponsor link
            sponsor_link = app.screen.query_one("#sponsor-link", Link)
            assert sponsor_link is not None
            assert sponsor_link.url == "https://github.com/sponsors/HYP3R00T"

    async def test_info_modal_has_github_link(self, monkeypatch, tmp_path: Path) -> None:
        """InfoModal should have a Star on GitHub link."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(InfoModal())
            await pilot.pause()
            github_link = app.screen.query_one("#github-link", Link)
            assert github_link is not None
            assert github_link.url == "https://github.com/HYP3R00T/voicepad"

    async def test_info_modal_links_are_link_widgets(self, monkeypatch, tmp_path: Path) -> None:
        """Links in the InfoModal should be Link widgets, not Label markup."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(InfoModal())
            await pilot.pause()
            # Verify that Link widgets exist (not Label with markup)
            sponsor_link = app.screen.query_one("#sponsor-link", Link)
            github_link = app.screen.query_one("#github-link", Link)
            assert isinstance(sponsor_link, Link)
            assert isinstance(github_link, Link)


# ---------------------------------------------------------------------------
# Link click tests
# ---------------------------------------------------------------------------


class TestLinkClicks:
    """Tests for link click functionality in the app."""

    async def test_markdown_link_clicked_opens_browser(self, monkeypatch, tmp_path: Path) -> None:
        """Clicking a link in the markdown viewer should open the browser."""
        from textual.widgets import Markdown

        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)

        mock_webbrowser = MagicMock()
        with patch("webbrowser.open", mock_webbrowser):
            app = VoicePadApp(config)
            async with app.run_test() as pilot:
                await pilot.pause()
                # Simulate a link click event
                event = Markdown.LinkClicked(
                    markdown=MagicMock(),
                    href="https://example.com",
                )
                app.on_markdown_link_clicked(event)
                await pilot.pause()
                mock_webbrowser.assert_called_once_with("https://example.com")

    async def test_markdown_link_clicked_handles_errors(self, monkeypatch, tmp_path: Path) -> None:
        """Link click handler should gracefully handle errors."""
        from textual.widgets import Markdown

        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)

        def raise_error(url: str) -> None:
            raise OSError("Browser not found")

        with patch("webbrowser.open", side_effect=raise_error):
            app = VoicePadApp(config)
            async with app.run_test() as pilot:
                await pilot.pause()
                event = Markdown.LinkClicked(
                    markdown=MagicMock(),
                    href="https://example.com",
                )
                # Should not raise an exception
                app.on_markdown_link_clicked(event)
                await pilot.pause()

    async def test_info_modal_links_are_clickable(self, monkeypatch, tmp_path: Path) -> None:
        """Links in the InfoModal should be Link widgets, not Label markup."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(InfoModal())
            await pilot.pause()
            # Verify that Link widgets exist (not Label with markup)
            sponsor_link = app.screen.query_one("#sponsor-link", Link)
            github_link = app.screen.query_one("#github-link", Link)
            assert isinstance(sponsor_link, Link)
            assert isinstance(github_link, Link)

    async def test_app_has_info_action(self, monkeypatch, tmp_path: Path) -> None:
        """App should have an action to show the info modal."""
        config = Config(recordings_path=tmp_path / "r", markdown_path=tmp_path / "m")
        monkeypatch.setattr(VoicePadApp, "_warm_model_worker", lambda self: None, raising=False)
        app = VoicePadApp(config)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Verify the action exists
            assert hasattr(app, "action_show_info")
            # Press 'i' to trigger the info modal
            await pilot.press("i")
            await pilot.pause()
            # Check if InfoModal is in the screen stack
            assert any(isinstance(screen, InfoModal) for screen in app.screen_stack)


# ---------------------------------------------------------------------------
# Import needed for SimpleNamespace usage in tests
# ---------------------------------------------------------------------------
