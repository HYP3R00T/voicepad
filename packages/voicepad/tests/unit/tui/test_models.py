"""Tests for VoicePad TUI models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from voicepad.tui.models import SessionEntry


class TestSessionEntry:
    """Test suite for SessionEntry dataclass."""

    def test_init_with_all_fields(self) -> None:
        """SessionEntry can be initialized with all fields."""
        entry = SessionEntry(
            index=0,
            wav_path=Path("/test/recording.wav"),
            md_path=Path("/test/recording.md"),
            duration_s=5.5,
            text="Test transcription",
            latency_ms=150.0,
            device="cuda",
            timestamp="2026-01-01 10:00",
        )

        assert entry.index == 0
        assert entry.wav_path == Path("/test/recording.wav")
        assert entry.md_path == Path("/test/recording.md")
        assert entry.duration_s == 5.5
        assert entry.text == "Test transcription"
        assert entry.latency_ms == 150.0
        assert entry.device == "cuda"
        assert entry.timestamp == "2026-01-01 10:00"

    def test_init_with_none_paths(self) -> None:
        """SessionEntry can have None for wav_path and md_path."""
        entry = SessionEntry(
            index=1,
            wav_path=None,
            md_path=None,
            duration_s=3.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
        )

        assert entry.wav_path is None
        assert entry.md_path is None

    @patch("voicepad.tui.models.time.strftime")
    def test_init_generates_default_timestamp(self, mock_strftime) -> None:
        """SessionEntry generates timestamp if not provided."""
        mock_strftime.return_value = "2026-01-15 14:30"

        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=0.0,
            text="",
            latency_ms=0.0,
            device="cpu",
        )

        assert entry.timestamp == "2026-01-15 14:30"
        mock_strftime.assert_called_once_with("%Y-%m-%d %H:%M")

    def test_fields_are_mutable(self) -> None:
        """SessionEntry fields can be modified after creation."""
        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=0.0,
            text="Original",
            latency_ms=0.0,
            device="cpu",
        )

        entry.text = "Modified"
        entry.duration_s = 10.0
        entry.device = "cuda"

        assert entry.text == "Modified"
        assert entry.duration_s == 10.0
        assert entry.device == "cuda"

    def test_equality(self) -> None:
        """SessionEntry instances with same values are equal."""
        entry1 = SessionEntry(
            index=0,
            wav_path=Path("/test.wav"),
            md_path=Path("/test.md"),
            duration_s=5.0,
            text="Test",
            latency_ms=100.0,
            device="cuda",
            timestamp="2026-01-01 10:00",
        )

        entry2 = SessionEntry(
            index=0,
            wav_path=Path("/test.wav"),
            md_path=Path("/test.md"),
            duration_s=5.0,
            text="Test",
            latency_ms=100.0,
            device="cuda",
            timestamp="2026-01-01 10:00",
        )

        assert entry1 == entry2

    def test_inequality(self) -> None:
        """SessionEntry instances with different values are not equal."""
        entry1 = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=5.0,
            text="Test1",
            latency_ms=100.0,
            device="cuda",
        )

        entry2 = SessionEntry(
            index=1,
            wav_path=None,
            md_path=None,
            duration_s=5.0,
            text="Test2",
            latency_ms=100.0,
            device="cuda",
        )

        assert entry1 != entry2

    def test_repr(self) -> None:
        """SessionEntry has a useful repr."""
        entry = SessionEntry(
            index=0,
            wav_path=Path("/test.wav"),
            md_path=Path("/test.md"),
            duration_s=5.0,
            text="Test",
            latency_ms=100.0,
            device="cuda",
            timestamp="2026-01-01 10:00",
        )

        repr_str = repr(entry)
        assert "SessionEntry" in repr_str
        assert "index=0" in repr_str
        assert "Test" in repr_str

    def test_handles_empty_text(self) -> None:
        """SessionEntry handles empty text string."""
        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=0.0,
            text="",
            latency_ms=0.0,
            device="cpu",
        )

        assert entry.text == ""

    def test_handles_zero_duration(self) -> None:
        """SessionEntry handles zero duration."""
        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=0.0,
            text="Test",
            latency_ms=0.0,
            device="cpu",
        )

        assert entry.duration_s == 0.0

    def test_handles_large_latency(self) -> None:
        """SessionEntry handles large latency values."""
        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=100.0,
            text="Test",
            latency_ms=50000.0,  # 50 seconds
            device="cpu",
        )

        assert entry.latency_ms == 50000.0
