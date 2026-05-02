"""Tests for HistoryService."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from voicepad.tui.models import SessionEntry
from voicepad.tui.services.history_service import HistoryService


def create_mock_config(tmp_path: Path) -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.markdown_path = tmp_path / "markdown"
    config.recordings_path = tmp_path / "recordings"
    return config


def create_mock_transcription_result() -> MagicMock:
    """Create a mock TranscriptionResult."""
    result = MagicMock()
    result.text = "Test transcription"
    result.latency_ms = 100.0
    result.device = "cpu"
    return result


def create_mock_chunk_result() -> MagicMock:
    """Create a mock ChunkResult."""
    chunk = MagicMock()
    chunk.text = "Test chunk"
    chunk.start = 0.0
    chunk.end = 1.0
    return chunk


class TestHistoryService:
    """Test suite for HistoryService."""

    def test_init_stores_config(self, tmp_path: Path) -> None:
        """HistoryService stores the config object."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)
        assert service.config == config

    def test_init_creates_empty_entries_list(self, tmp_path: Path) -> None:
        """HistoryService initializes with empty entries list."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)
        assert service.entries == []

    def test_load_from_disk_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        """load_from_disk returns empty list when markdown directory doesn't exist."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        entries = service.load_from_disk()

        assert entries == []
        assert service.entries == []

    @patch("voicepad.tui.services.history_service.parse_markdown_entry")
    def test_load_from_disk_loads_markdown_files(self, mock_parse: MagicMock, tmp_path: Path) -> None:
        """load_from_disk loads all markdown files from disk."""
        config = create_mock_config(tmp_path)
        config.markdown_path.mkdir(parents=True)

        # Create test markdown files
        (config.markdown_path / "test1.md").write_text("# Test 1")
        (config.markdown_path / "test2.md").write_text("# Test 2")

        # Mock parse_markdown_entry to return SessionEntry objects
        entry1 = SessionEntry(
            index=0,
            wav_path=None,
            md_path=config.markdown_path / "test1.md",
            duration_s=1.0,
            text="Test 1",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        entry2 = SessionEntry(
            index=1,
            wav_path=None,
            md_path=config.markdown_path / "test2.md",
            duration_s=2.0,
            text="Test 2",
            latency_ms=200.0,
            device="cpu",
            timestamp="2024-01-01 12:01:00",
        )
        mock_parse.side_effect = [entry1, entry2]

        service = HistoryService(config)
        entries = service.load_from_disk()

        assert len(entries) == 2
        assert entries[0] == entry1
        assert entries[1] == entry2
        assert service.entries == entries

    @patch("voicepad.tui.services.history_service.parse_markdown_entry")
    def test_load_from_disk_skips_none_entries(self, mock_parse: MagicMock, tmp_path: Path) -> None:
        """load_from_disk skips entries that parse_markdown_entry returns None for."""
        config = create_mock_config(tmp_path)
        config.markdown_path.mkdir(parents=True)

        (config.markdown_path / "test1.md").write_text("# Test 1")
        (config.markdown_path / "test2.md").write_text("# Invalid")

        entry1 = SessionEntry(
            index=0,
            wav_path=None,
            md_path=config.markdown_path / "test1.md",
            duration_s=1.0,
            text="Test 1",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        mock_parse.side_effect = [entry1, None]

        service = HistoryService(config)
        entries = service.load_from_disk()

        assert len(entries) == 1
        assert entries[0] == entry1

    def test_load_from_disk_clears_existing_entries(self, tmp_path: Path) -> None:
        """load_from_disk clears existing entries before loading."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        # Add some existing entries
        service.entries = [
            SessionEntry(
                index=0,
                wav_path=None,
                md_path=None,
                duration_s=1.0,
                text="Old entry",
                latency_ms=100.0,
                device="cpu",
                timestamp="2024-01-01 12:00:00",
            )
        ]

        # Load from disk (empty directory)
        entries = service.load_from_disk()

        assert entries == []
        assert service.entries == []

    def test_add_entry_appends_to_list(self, tmp_path: Path) -> None:
        """add_entry appends entry to the entries list."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )

        service.add_entry(entry)

        assert len(service.entries) == 1
        assert service.entries[0] == entry

    def test_add_entry_maintains_order(self, tmp_path: Path) -> None:
        """add_entry maintains insertion order."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        entry1 = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=1.0,
            text="First",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        entry2 = SessionEntry(
            index=1,
            wav_path=None,
            md_path=None,
            duration_s=2.0,
            text="Second",
            latency_ms=200.0,
            device="cpu",
            timestamp="2024-01-01 12:01:00",
        )

        service.add_entry(entry1)
        service.add_entry(entry2)

        assert service.entries[0] == entry1
        assert service.entries[1] == entry2

    def test_get_entry_returns_entry_by_index(self, tmp_path: Path) -> None:
        """get_entry returns the correct entry by index."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        service.entries.append(entry)

        result = service.get_entry(0)

        assert result == entry

    def test_get_entry_returns_none_for_invalid_index(self, tmp_path: Path) -> None:
        """get_entry returns None for invalid index."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        assert service.get_entry(0) is None
        assert service.get_entry(-1) is None
        assert service.get_entry(999) is None

    def test_delete_entry_removes_from_list(self, tmp_path: Path) -> None:
        """delete_entry removes entry from the list."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        service.entries.append(entry)

        result = service.delete_entry(0)

        assert result is True
        assert len(service.entries) == 0

    def test_delete_entry_deletes_wav_file(self, tmp_path: Path) -> None:
        """delete_entry deletes the WAV file if it exists."""
        config = create_mock_config(tmp_path)
        config.recordings_path.mkdir(parents=True)
        service = HistoryService(config)

        wav_path = config.recordings_path / "test.wav"
        wav_path.write_text("fake wav")

        entry = SessionEntry(
            index=0,
            wav_path=wav_path,
            md_path=None,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        service.entries.append(entry)

        service.delete_entry(0)

        assert not wav_path.exists()

    def test_delete_entry_deletes_markdown_file(self, tmp_path: Path) -> None:
        """delete_entry deletes the markdown file if it exists."""
        config = create_mock_config(tmp_path)
        config.markdown_path.mkdir(parents=True)
        service = HistoryService(config)

        md_path = config.markdown_path / "test.md"
        md_path.write_text("# Test")

        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=md_path,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        service.entries.append(entry)

        service.delete_entry(0)

        assert not md_path.exists()

    def test_delete_entry_reindexes_remaining_entries(self, tmp_path: Path) -> None:
        """delete_entry re-indexes remaining entries after deletion."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        entry1 = SessionEntry(
            index=0,
            wav_path=None,
            md_path=None,
            duration_s=1.0,
            text="First",
            latency_ms=100.0,
            device="cpu",
            timestamp="2024-01-01 12:00:00",
        )
        entry2 = SessionEntry(
            index=1,
            wav_path=None,
            md_path=None,
            duration_s=2.0,
            text="Second",
            latency_ms=200.0,
            device="cpu",
            timestamp="2024-01-01 12:01:00",
        )
        entry3 = SessionEntry(
            index=2,
            wav_path=None,
            md_path=None,
            duration_s=3.0,
            text="Third",
            latency_ms=300.0,
            device="cpu",
            timestamp="2024-01-01 12:02:00",
        )
        service.entries = [entry1, entry2, entry3]

        service.delete_entry(1)

        assert len(service.entries) == 2
        assert service.entries[0].index == 0
        assert service.entries[0].text == "First"
        assert service.entries[1].index == 1
        assert service.entries[1].text == "Third"

    def test_delete_entry_returns_false_for_invalid_index(self, tmp_path: Path) -> None:
        """delete_entry returns False for invalid index."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        assert service.delete_entry(0) is False
        assert service.delete_entry(-1) is False
        assert service.delete_entry(999) is False

    @patch("voicepad.tui.services.history_service.format_markdown")
    def test_save_markdown_creates_directory(self, mock_format: MagicMock, tmp_path: Path) -> None:
        """save_markdown creates markdown directory if it doesn't exist."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        wav_path = tmp_path / "test.wav"
        result = create_mock_transcription_result()
        mock_format.return_value = "# Test"

        service.save_markdown(wav_path, result, "turbo")

        assert config.markdown_path.exists()

    @patch("voicepad.tui.services.history_service.format_markdown")
    def test_save_markdown_writes_file(self, mock_format: MagicMock, tmp_path: Path) -> None:
        """save_markdown writes markdown content to file."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        wav_path = tmp_path / "test.wav"
        result = create_mock_transcription_result()
        mock_format.return_value = "# Test Content"

        md_path = service.save_markdown(wav_path, result, "turbo")

        assert md_path.exists()
        assert md_path.read_text(encoding="utf-8") == "# Test Content"

    @patch("voicepad.tui.services.history_service.format_markdown")
    def test_save_markdown_returns_path(self, mock_format: MagicMock, tmp_path: Path) -> None:
        """save_markdown returns the path to the created file."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        wav_path = tmp_path / "test.wav"
        result = create_mock_transcription_result()
        mock_format.return_value = "# Test"

        md_path = service.save_markdown(wav_path, result, "turbo")

        expected_path = config.markdown_path / "test.md"
        assert md_path == expected_path

    @patch("voicepad.tui.services.history_service.format_markdown_streaming")
    def test_save_markdown_streaming_creates_directory(self, mock_format: MagicMock, tmp_path: Path) -> None:
        """save_markdown_streaming creates markdown directory if it doesn't exist."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        wav_path = tmp_path / "test.wav"
        chunks = [create_mock_chunk_result()]
        mock_format.return_value = "# Test"

        service.save_markdown_streaming(wav_path, "Test text", 1.0, cast(list, chunks), "turbo")

        assert config.markdown_path.exists()

    @patch("voicepad.tui.services.history_service.format_markdown_streaming")
    def test_save_markdown_streaming_writes_file(self, mock_format: MagicMock, tmp_path: Path) -> None:
        """save_markdown_streaming writes markdown content to file."""
        config = create_mock_config(tmp_path)
        service = HistoryService(config)

        wav_path = tmp_path / "test.wav"
        chunks = [create_mock_chunk_result()]
        mock_format.return_value = "# Streaming Content"

        md_path = service.save_markdown_streaming(wav_path, "Test text", 1.0, cast(list, chunks), "turbo")

        assert md_path.exists()
        assert md_path.read_text(encoding="utf-8") == "# Streaming Content"

    @patch("voicepad.tui.services.history_service.prepend_retranscription")
    def test_update_markdown_with_retranscription_updates_file(self, mock_prepend: MagicMock, tmp_path: Path) -> None:
        """update_markdown_with_retranscription updates the markdown file."""
        config = create_mock_config(tmp_path)
        config.markdown_path.mkdir(parents=True)
        service = HistoryService(config)

        md_path = config.markdown_path / "test.md"
        md_path.write_text("# Original")

        result = create_mock_transcription_result()
        mock_prepend.return_value = "# Updated Content"

        service.update_markdown_with_retranscription(md_path, result, "turbo")

        assert md_path.read_text(encoding="utf-8") == "# Updated Content"

    @patch("voicepad.tui.services.history_service.prepend_retranscription")
    def test_update_markdown_with_retranscription_calls_prepend(self, mock_prepend: MagicMock, tmp_path: Path) -> None:
        """update_markdown_with_retranscription calls prepend_retranscription."""
        config = create_mock_config(tmp_path)
        config.markdown_path.mkdir(parents=True)
        service = HistoryService(config)

        md_path = config.markdown_path / "test.md"
        md_path.write_text("# Original")

        result = create_mock_transcription_result()
        mock_prepend.return_value = "# Updated"

        service.update_markdown_with_retranscription(md_path, result, "turbo")

        mock_prepend.assert_called_once_with(md_path, result, "turbo")
