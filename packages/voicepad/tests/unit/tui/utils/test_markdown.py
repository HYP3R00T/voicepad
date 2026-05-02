"""Tests for markdown utilities."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from voicepad.tui.utils.markdown import (
    format_markdown,
    format_markdown_streaming,
    parse_markdown_entry,
    prepend_retranscription,
)


class TestFormatMarkdown:
    """Test suite for format_markdown function."""

    def test_format_markdown_creates_basic_structure(self) -> None:
        """format_markdown creates markdown with basic structure."""
        audio_path = Path("/test/recording.wav")
        result = MagicMock()
        result.device = "cuda"
        result.compute_type = "float16"
        result.language = "en"
        result.language_probability = 0.95
        result.duration_s = 5.5
        result.latency_ms = 150.0
        result.text = "Hello world"
        result.segments = []

        markdown = format_markdown(audio_path, result, "turbo")

        assert "---" in markdown
        assert "file: recording.wav" in markdown
        assert "transcriptions:" in markdown
        assert "- n: 1" in markdown
        assert "model: turbo · cuda / float16" in markdown
        assert "language: en (95.0%)" in markdown
        assert "duration: 5.5s" in markdown
        assert "latency: 150ms" in markdown
        assert "## Transcription 1" in markdown
        assert "Hello world" in markdown

    def test_format_markdown_handles_no_speech(self) -> None:
        """format_markdown handles empty text as no speech detected."""
        audio_path = Path("/test/recording.wav")
        result = MagicMock()
        result.device = "cpu"
        result.compute_type = "int8"
        result.language = "en"
        result.language_probability = 1.0
        result.duration_s = 2.0
        result.latency_ms = 100.0
        result.text = ""
        result.segments = []

        markdown = format_markdown(audio_path, result)

        assert "*(no speech detected)*" in markdown

    def test_format_markdown_includes_segments(self) -> None:
        """format_markdown includes segment information when available."""
        audio_path = Path("/test/recording.wav")
        result = MagicMock()
        result.device = "cuda"
        result.compute_type = "float16"
        result.language = "en"
        result.language_probability = 0.98
        result.duration_s = 10.0
        result.latency_ms = 200.0
        result.text = "First segment. Second segment."

        seg1 = MagicMock()
        seg1.start = 0.0
        seg1.end = 5.0
        seg1.text = "First segment."

        seg2 = MagicMock()
        seg2.start = 5.0
        seg2.end = 10.0
        seg2.text = "Second segment."

        result.segments = [seg1, seg2]

        markdown = format_markdown(audio_path, result, "base")

        assert "## Segments" in markdown
        assert "**0.0s – 5.0s** First segment." in markdown
        assert "**5.0s – 10.0s** Second segment." in markdown

    def test_format_markdown_without_model_name(self) -> None:
        """format_markdown works without model name."""
        audio_path = Path("/test/recording.wav")
        result = MagicMock()
        result.device = "cpu"
        result.compute_type = "int8"
        result.language = "en"
        result.language_probability = 1.0
        result.duration_s = 3.0
        result.latency_ms = 50.0
        result.text = "Test"
        result.segments = []

        markdown = format_markdown(audio_path, result)

        assert "model: cpu / int8" in markdown
        assert "turbo" not in markdown

    def test_format_markdown_handles_missing_attributes(self) -> None:
        """format_markdown handles result objects with missing attributes."""
        audio_path = Path("/test/recording.wav")
        result = MagicMock()
        # Set all required attributes with proper types
        result.device = "unknown"
        result.compute_type = "unknown"
        result.language = "en"
        result.language_probability = 1.0
        result.duration_s = 0.0
        result.latency_ms = 0.0
        result.text = "Test transcription"
        result.segments = []

        markdown = format_markdown(audio_path, result)

        assert "file: recording.wav" in markdown
        assert "Test transcription" in markdown
        assert "unknown" in markdown  # Default device


class TestFormatMarkdownStreaming:
    """Test suite for format_markdown_streaming function."""

    def test_format_markdown_streaming_creates_basic_structure(self) -> None:
        """format_markdown_streaming creates markdown with streaming info."""
        wav_path = Path("/test/recording.wav")
        text = "Streaming transcription"
        duration_s = 8.5

        chunk = MagicMock()
        chunk.text = "Streaming transcription"
        chunk.device = "cuda"
        chunk.language = "en"
        chunk.language_probability = 0.96
        chunk.latency_ms = 100.0
        chunk.segments = []

        chunks = [chunk]

        markdown = format_markdown_streaming(wav_path, text, duration_s, cast(list, chunks), "turbo")

        assert "---" in markdown
        assert "file: recording.wav" in markdown
        assert "model: turbo · cuda / streaming" in markdown
        assert "language: en (96.0%)" in markdown
        assert "duration: 8.5s" in markdown
        assert "latency: 100ms" in markdown
        assert "## Transcription 1" in markdown
        assert "Streaming transcription" in markdown

    def test_format_markdown_streaming_sums_latencies(self) -> None:
        """format_markdown_streaming sums latencies from all chunks."""
        wav_path = Path("/test/recording.wav")
        text = "Combined text"
        duration_s = 10.0

        chunk1 = MagicMock()
        chunk1.text = "First"
        chunk1.device = "cuda"
        chunk1.language = "en"
        chunk1.language_probability = 0.95
        chunk1.latency_ms = 50.0
        chunk1.segments = []

        chunk2 = MagicMock()
        chunk2.text = "Second"
        chunk2.device = "cuda"
        chunk2.language = "en"
        chunk2.language_probability = 0.97
        chunk2.latency_ms = 75.0
        chunk2.segments = []

        chunks = [chunk1, chunk2]

        markdown = format_markdown_streaming(wav_path, text, duration_s, cast(list, chunks))

        assert "latency: 125ms" in markdown  # 50 + 75

    def test_format_markdown_streaming_handles_empty_text(self) -> None:
        """format_markdown_streaming handles empty text."""
        wav_path = Path("/test/recording.wav")
        text = ""
        duration_s = 5.0

        chunk = MagicMock()
        chunk.text = ""
        chunk.device = "cpu"
        chunk.language = "en"
        chunk.language_probability = 1.0
        chunk.latency_ms = 50.0
        chunk.segments = []

        chunks = [chunk]

        markdown = format_markdown_streaming(wav_path, text, duration_s, cast(list, chunks))

        assert "*(no speech detected)*" in markdown

    def test_format_markdown_streaming_includes_all_segments(self) -> None:
        """format_markdown_streaming includes segments from all chunks."""
        wav_path = Path("/test/recording.wav")
        text = "Combined segments"
        duration_s = 10.0

        seg1 = MagicMock()
        seg1.start = 0.0
        seg1.end = 3.0
        seg1.text = "First"

        seg2 = MagicMock()
        seg2.start = 3.0
        seg2.end = 6.0
        seg2.text = "Second"

        chunk1 = MagicMock()
        chunk1.text = "First"
        chunk1.device = "cuda"
        chunk1.language = "en"
        chunk1.language_probability = 0.95
        chunk1.latency_ms = 50.0
        chunk1.segments = [seg1]

        chunk2 = MagicMock()
        chunk2.text = "Second"
        chunk2.device = "cuda"
        chunk2.language = "en"
        chunk2.language_probability = 0.96
        chunk2.latency_ms = 60.0
        chunk2.segments = [seg2]

        chunks = [chunk1, chunk2]

        markdown = format_markdown_streaming(wav_path, text, duration_s, cast(list, chunks))

        assert "## Segments" in markdown
        assert "**0.0s – 3.0s** First" in markdown
        assert "**3.0s – 6.0s** Second" in markdown

    def test_format_markdown_streaming_uses_latest_chunk_metadata(self) -> None:
        """format_markdown_streaming uses metadata from latest non-empty chunk."""
        wav_path = Path("/test/recording.wav")
        text = "Test"
        duration_s = 5.0

        chunk1 = MagicMock()
        chunk1.text = "Old"
        chunk1.device = "cpu"
        chunk1.language = "en"
        chunk1.language_probability = 0.90
        chunk1.latency_ms = 50.0
        chunk1.segments = []

        chunk2 = MagicMock()
        chunk2.text = "Latest"
        chunk2.device = "cuda"
        chunk2.language = "es"
        chunk2.language_probability = 0.98
        chunk2.latency_ms = 60.0
        chunk2.segments = []

        chunks = [chunk1, chunk2]

        markdown = format_markdown_streaming(wav_path, text, duration_s, cast(list, chunks))

        # Should use latest chunk's metadata
        assert "cuda" in markdown
        assert "language: es (98.0%)" in markdown


class TestPrependRetranscription:
    """Test suite for prepend_retranscription function."""

    def test_prepend_retranscription_adds_new_entry(self, tmp_path: Path) -> None:
        """prepend_retranscription adds new transcription entry."""
        md_path = tmp_path / "test.md"
        existing_content = """---
file: test.wav
transcriptions:
  - n: 1
    model: turbo · cuda / float16
    language: en (95.0%)
    duration: 5.0s
    latency: 100ms
    timestamp: 2026-01-01 10:00
---

## Transcription 1

Original text
"""
        md_path.write_text(existing_content, encoding="utf-8")

        result = MagicMock()
        result.device = "cuda"
        result.compute_type = "float16"
        result.language = "en"
        result.language_probability = 0.97
        result.duration_s = 6.0
        result.latency_ms = 120.0
        result.text = "New transcription"

        new_content = prepend_retranscription(md_path, result, "turbo")

        assert "- n: 2" in new_content
        assert "- n: 1" in new_content
        assert "## Transcription 2" in new_content
        assert "New transcription" in new_content
        assert "## Transcription 1" in new_content
        assert "Original text" in new_content

    def test_prepend_retranscription_increments_n(self, tmp_path: Path) -> None:
        """prepend_retranscription correctly increments n value."""
        md_path = tmp_path / "test.md"
        existing_content = """---
file: test.wav
transcriptions:
  - n: 3
    model: base · cpu / int8
    language: en (90.0%)
    duration: 4.0s
    latency: 80ms
    timestamp: 2026-01-01 12:00
  - n: 2
    model: turbo · cuda / float16
    language: en (95.0%)
    duration: 5.0s
    latency: 100ms
    timestamp: 2026-01-01 11:00
  - n: 1
    model: turbo · cuda / float16
    language: en (93.0%)
    duration: 5.5s
    latency: 110ms
    timestamp: 2026-01-01 10:00
---

## Transcription 3

Third text

## Transcription 2

Second text

## Transcription 1

First text
"""
        md_path.write_text(existing_content, encoding="utf-8")

        result = MagicMock()
        result.device = "cuda"
        result.compute_type = "float16"
        result.language = "en"
        result.language_probability = 0.98
        result.duration_s = 7.0
        result.latency_ms = 130.0
        result.text = "Fourth transcription"

        new_content = prepend_retranscription(md_path, result, "turbo")

        assert "- n: 4" in new_content
        assert "## Transcription 4" in new_content

    def test_prepend_retranscription_handles_missing_file(self, tmp_path: Path) -> None:
        """prepend_retranscription creates new file if missing."""
        md_path = tmp_path / "nonexistent.md"

        result = MagicMock()
        result.device = "cpu"
        result.compute_type = "int8"
        result.language = "en"
        result.language_probability = 1.0
        result.duration_s = 3.0
        result.latency_ms = 50.0
        result.text = "First transcription"

        new_content = prepend_retranscription(md_path, result)

        assert "- n: 1" in new_content
        assert "file: nonexistent.wav" in new_content
        assert "## Transcription 1" in new_content
        assert "First transcription" in new_content

    def test_prepend_retranscription_handles_empty_text(self, tmp_path: Path) -> None:
        """prepend_retranscription handles empty transcription text."""
        md_path = tmp_path / "test.md"
        md_path.write_text("---\nfile: test.wav\ntranscriptions:\n---\n", encoding="utf-8")

        result = MagicMock()
        result.device = "cpu"
        result.compute_type = "int8"
        result.language = "en"
        result.language_probability = 1.0
        result.duration_s = 2.0
        result.latency_ms = 40.0
        result.text = ""

        new_content = prepend_retranscription(md_path, result)

        assert "*(no speech detected)*" in new_content


class TestParseMarkdownEntry:
    """Test suite for parse_markdown_entry function."""

    def test_parse_markdown_entry_returns_session_entry(self, tmp_path: Path) -> None:
        """parse_markdown_entry returns SessionEntry with correct data."""
        md_path = tmp_path / "test.md"
        content = """---
file: recording.wav
transcriptions:
  - n: 1
    model: turbo · cuda / float16
    language: en (95.0%)
    duration: 5.5s
    latency: 150ms
    timestamp: 2026-01-01 10:00
---

## Transcription 1

Hello world test transcription
"""
        md_path.write_text(content, encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0)

        assert entry is not None
        assert entry.index == 0
        assert entry.duration_s == 5.5
        assert entry.latency_ms == 150.0
        assert entry.device == "turbo · cuda"
        assert entry.text == "Hello world test transcription"
        assert entry.timestamp == "2026-01-01 10:00"

    def test_parse_markdown_entry_uses_latest_transcription(self, tmp_path: Path) -> None:
        """parse_markdown_entry uses the latest (highest n) transcription."""
        md_path = tmp_path / "test.md"
        content = """---
file: recording.wav
transcriptions:
  - n: 2
    model: base · cpu / int8
    language: en (90.0%)
    duration: 6.0s
    latency: 200ms
    timestamp: 2026-01-01 11:00
  - n: 1
    model: turbo · cuda / float16
    language: en (95.0%)
    duration: 5.5s
    latency: 150ms
    timestamp: 2026-01-01 10:00
---

## Transcription 2

Latest transcription text

## Transcription 1

Original transcription text
"""
        md_path.write_text(content, encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0)

        assert entry is not None
        assert entry.text == "Latest transcription text"
        assert entry.duration_s == 6.0
        assert entry.latency_ms == 200.0
        assert entry.device == "base · cpu"

    def test_parse_markdown_entry_finds_wav_file(self, tmp_path: Path) -> None:
        """parse_markdown_entry finds wav file in recordings directory."""
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()
        wav_path = recordings_dir / "test.wav"
        wav_path.write_bytes(b"fake wav data")

        markdown_dir = tmp_path / "markdown"
        markdown_dir.mkdir()
        md_path = markdown_dir / "test.md"
        content = """---
file: test.wav
transcriptions:
  - n: 1
    model: turbo · cuda / float16
    language: en (95.0%)
    duration: 5.0s
    latency: 100ms
    timestamp: 2026-01-01 10:00
---

## Transcription 1

Test text
"""
        md_path.write_text(content, encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0, recordings_path=recordings_dir)

        assert entry is not None
        assert entry.wav_path == wav_path

    def test_parse_markdown_entry_returns_none_for_invalid_file(self, tmp_path: Path) -> None:
        """parse_markdown_entry returns None for invalid markdown."""
        md_path = tmp_path / "invalid.md"
        md_path.write_text("Not valid markdown", encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0)

        assert entry is None

    def test_parse_markdown_entry_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """parse_markdown_entry returns None for missing file."""
        md_path = tmp_path / "nonexistent.md"

        entry = parse_markdown_entry(md_path, 0)

        assert entry is None

    def test_parse_markdown_entry_returns_none_for_empty_text(self, tmp_path: Path) -> None:
        """parse_markdown_entry returns None if text is empty."""
        md_path = tmp_path / "test.md"
        content = """---
file: test.wav
transcriptions:
  - n: 1
    model: turbo · cuda / float16
    language: en (95.0%)
    duration: 5.0s
    latency: 100ms
    timestamp: 2026-01-01 10:00
---

## Transcription 1

*(no speech detected)*
"""
        md_path.write_text(content, encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0)

        assert entry is None

    def test_parse_markdown_entry_extracts_timestamp_from_filename(self, tmp_path: Path) -> None:
        """parse_markdown_entry extracts timestamp from filename if missing."""
        md_path = tmp_path / "recording_20260115_1430.md"
        content = """---
file: recording.wav
transcriptions:
  - n: 1
    model: turbo · cuda / float16
    language: en (95.0%)
    duration: 5.0s
    latency: 100ms
---

## Transcription 1

Test text
"""
        md_path.write_text(content, encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0)

        assert entry is not None
        assert entry.timestamp == "2026-01-15 14:30"

    def test_parse_markdown_entry_handles_malformed_front_matter(self, tmp_path: Path) -> None:
        """parse_markdown_entry handles malformed front matter gracefully."""
        md_path = tmp_path / "test.md"
        content = """---
file: test.wav
transcriptions:
  - n: 1
    model: turbo
    duration: invalid
    latency: also_invalid
---

## Transcription 1

Test text
"""
        md_path.write_text(content, encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0)

        # Should still parse with default values
        assert entry is not None
        assert entry.text == "Test text"
        assert entry.duration_s == 0.0
        assert entry.latency_ms == 0.0
