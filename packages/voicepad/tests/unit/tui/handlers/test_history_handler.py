"""Tests for HistoryHandler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from voicepad.tui.handlers.history_handler import HistoryHandler
from voicepad.tui.models import SessionEntry


class TestHistoryHandlerInit:
    """Tests for HistoryHandler initialization."""

    def test_init_stores_app_reference(self):
        """Test that __init__ stores the app reference."""
        mock_app = Mock()
        handler = HistoryHandler(mock_app)
        assert handler.app is mock_app


class TestLoadHistoryFromDisk:
    """Tests for load_history_from_disk method."""

    def test_returns_early_when_markdown_dir_missing(self, tmp_path):
        """Test that method returns early when markdown directory doesn't exist."""
        mock_app = Mock()
        mock_app.config.markdown_path = tmp_path / "nonexistent"
        mock_app._entries = []

        handler = HistoryHandler(mock_app)
        handler.load_history_from_disk()

        assert len(mock_app._entries) == 0

    def test_loads_markdown_files_from_disk(self, tmp_path):
        """Test that method loads markdown files from disk."""
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        # Create a test markdown file
        md_file = md_dir / "test.md"
        md_file.write_text("---\nfile: test.wav\n---\n\nTest content", encoding="utf-8")

        mock_app = Mock()
        mock_app.config.markdown_path = md_dir
        mock_app.config.recordings_path = tmp_path / "recordings"
        mock_app._entries = []

        # Mock the query_one to return a proper mock for OptionList
        mock_ol = Mock()
        mock_ol.option_count = 0
        mock_app.query_one.return_value = mock_ol

        handler = HistoryHandler(mock_app)

        with patch("voicepad.tui.handlers.history_handler._parse_markdown_entry") as mock_parse:
            mock_entry = SessionEntry(
                index=0,
                wav_path=None,
                md_path=md_file,
                duration_s=1.0,
                text="Test",
                latency_ms=100.0,
                device="cpu",
            )
            mock_parse.return_value = mock_entry

            handler.load_history_from_disk()

            assert len(mock_app._entries) == 1
            assert mock_app._entries[0] == mock_entry
            mock_ol.add_option.assert_called_once()


class TestAddHistoryEntry:
    """Tests for add_history_entry method."""

    def test_adds_entry_to_option_list(self):
        """Test that method refreshes the option list with existing entries."""
        mock_app = Mock()
        mock_ol = Mock()
        mock_ol.option_count = 1  # After refresh
        mock_app.query_one.return_value = mock_ol

        entry = SessionEntry(
            index=0,
            wav_path=Path("test.wav"),
            md_path=None,
            duration_s=1.5,
            text="Test",
            latency_ms=150.0,
            device="cuda",
        )

        # Entry must already be in _entries when add_history_entry is called
        mock_app._entries = [entry]
        mock_app._sort_ascending = False
        mock_app._selected_entry_idx = None

        handler = HistoryHandler(mock_app)
        handler.add_history_entry(entry)

        # refresh_history_list should have cleared and re-added the option
        mock_ol.clear_options.assert_called_once()
        mock_ol.add_option.assert_called_once()
        # After adding, highlighted should be set to option_count - 1 (which is 0)
        assert mock_ol.highlighted == 0


class TestToggleSortOrder:
    """Tests for toggle_sort_order method."""

    def test_shows_transient_sorted_status_on_toggle(self):
        """Test that sort toggle shows a temporary sorted status message."""
        entry = SessionEntry(
            index=0,
            wav_path=Path("test.wav"),
            md_path=None,
            duration_s=1.5,
            text="Test",
            latency_ms=150.0,
            device="cuda",
        )

        mock_app = Mock()
        mock_app._entries = [entry]
        mock_app._sort_ascending = True
        mock_app._selected_entry_idx = None
        mock_ol = Mock()
        mock_ol.option_count = 1
        mock_app.query_one.return_value = mock_ol

        handler = HistoryHandler(mock_app)
        handler.toggle_sort_order()

        # Sort direction toggles and status is shown only as a transient notice.
        assert mock_app._sort_ascending is False
        mock_app._set_status.assert_called_once_with("ready", "Sorted ↑")
        mock_app.set_timer.assert_called_once()

    def test_refresh_history_list_does_not_update_status(self):
        """Test that generic history refresh does not emit a sort status message."""
        entry = SessionEntry(
            index=0,
            wav_path=Path("test.wav"),
            md_path=None,
            duration_s=1.5,
            text="Test",
            latency_ms=150.0,
            device="cuda",
        )

        mock_app = Mock()
        mock_app._entries = [entry]
        mock_app._sort_ascending = True
        mock_app._selected_entry_idx = None
        mock_ol = Mock()
        mock_ol.option_count = 1
        mock_app.query_one.return_value = mock_ol

        handler = HistoryHandler(mock_app)
        handler.refresh_history_list()

        mock_app._set_status.assert_not_called()


class TestActionRetranscribeEntry:
    """Tests for action_retranscribe_entry method."""

    def test_returns_early_when_no_entry_selected(self):
        """Test that method returns early when no entry is selected."""
        mock_app = Mock()
        mock_app._selected_entry_idx = None
        mock_app._model_ready = True

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "retranscribe_file") as mock_retranscribe:
            handler.action_retranscribe_entry()
            mock_retranscribe.assert_not_called()

    def test_returns_early_when_model_not_ready(self):
        """Test that method returns early when model is not ready."""
        mock_app = Mock()
        mock_app._selected_entry_idx = 0
        mock_app._model_ready = False
        mock_app._entries = [Mock()]

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "retranscribe_file") as mock_retranscribe:
            handler.action_retranscribe_entry()
            mock_retranscribe.assert_not_called()

    def test_calls_retranscribe_when_conditions_met(self, tmp_path):
        """Test that method calls retranscribe_file when conditions are met."""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        entry = SessionEntry(
            index=0,
            wav_path=wav_path,
            md_path=None,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
        )

        mock_app = Mock()
        mock_app._selected_entry_idx = 0
        mock_app._model_ready = True
        mock_app._entries = [entry]

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "retranscribe_file") as mock_retranscribe:
            handler.action_retranscribe_entry()
            mock_retranscribe.assert_called_once_with(wav_path, None)


class TestActionOpenEntry:
    """Tests for open-entry actions."""

    def test_opens_selected_recording(self, tmp_path):
        """Test that the selected recording is opened with the system handler."""
        wav_path = tmp_path / "test.wav"
        wav_path.touch()

        entry = SessionEntry(
            index=0,
            wav_path=wav_path,
            md_path=None,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
        )

        mock_app = Mock()
        mock_app._selected_entry_idx = 0
        mock_app._entries = [entry]

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "_open_path") as mock_open:
            handler.action_open_recording()
            mock_open.assert_called_once_with(wav_path)

    def test_opens_selected_markdown(self, tmp_path):
        """Test that the selected markdown file is opened with the system handler."""
        md_path = tmp_path / "test.md"
        md_path.touch()

        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=md_path,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
        )

        mock_app = Mock()
        mock_app._selected_entry_idx = 0
        mock_app._entries = [entry]

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "_open_path") as mock_open:
            handler.action_open_markdown()
            mock_open.assert_called_once_with(md_path)

    def test_falls_back_to_browser_uri_when_native_open_fails(self, tmp_path):
        """Test that a browser URI fallback is attempted if the native opener fails."""
        md_path = tmp_path / "test.md"
        md_path.touch()

        entry = SessionEntry(
            index=0,
            wav_path=None,
            md_path=md_path,
            duration_s=1.0,
            text="Test",
            latency_ms=100.0,
            device="cpu",
        )

        mock_app = Mock()
        mock_app._selected_entry_idx = 0
        mock_app._entries = [entry]
        mock_app._set_status = Mock()

        handler = HistoryHandler(mock_app)

        with (
            patch("voicepad.tui.handlers.history_handler.sys.platform", "linux"),
            patch("subprocess.Popen", side_effect=FileNotFoundError),
            patch("voicepad.tui.handlers.history_handler.webbrowser.open", return_value=True) as mock_open,
        ):
            handler.action_open_markdown()

        mock_open.assert_called_once()
        mock_app._set_status.assert_not_called()


class TestActionDeleteEntry:
    """Tests for action_delete_entry method."""

    def test_returns_early_when_not_on_history_tab(self):
        """Test that method returns early when not on history tab."""
        mock_app = Mock()
        mock_tabbed = Mock()
        mock_tabbed.active = "tab-record"
        mock_app.query_one.return_value = mock_tabbed
        mock_app._selected_entry_idx = 0

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "show_delete_confirm") as mock_show:
            handler.action_delete_entry()
            mock_show.assert_not_called()

    def test_returns_early_when_no_entry_selected(self):
        """Test that method returns early when no entry is selected."""
        mock_app = Mock()
        mock_tabbed = Mock()
        mock_tabbed.active = "tab-history"
        mock_app.query_one.return_value = mock_tabbed
        mock_app._selected_entry_idx = None

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "show_delete_confirm") as mock_show:
            handler.action_delete_entry()
            mock_show.assert_not_called()

    def test_shows_delete_confirm_when_conditions_met(self):
        """Test that method shows delete confirmation when conditions are met."""
        mock_app = Mock()
        mock_tabbed = Mock()
        mock_tabbed.active = "tab-history"
        mock_app.query_one.return_value = mock_tabbed
        mock_app._selected_entry_idx = 0

        handler = HistoryHandler(mock_app)

        with patch.object(handler, "show_delete_confirm") as mock_show:
            handler.action_delete_entry()
            mock_show.assert_called_once()


class TestActionCopyTranscription:
    """Tests for action_copy_transcription method."""

    def test_returns_early_when_no_text(self):
        """Test that method returns early when there's no current text."""
        mock_app = Mock()
        mock_app._current_text = ""

        handler = HistoryHandler(mock_app)

        with patch("voicepad.tui.handlers.history_handler._copy_to_clipboard") as mock_copy:
            handler.action_copy_transcription()
            mock_copy.assert_not_called()

    def test_copies_text_to_clipboard(self):
        """Test that method copies text to clipboard."""
        mock_app = Mock()
        mock_app._current_text = "Test transcription"
        mock_btn = Mock()
        mock_app.query_one.return_value = mock_btn

        handler = HistoryHandler(mock_app)

        with patch("voicepad.tui.handlers.history_handler._copy_to_clipboard") as mock_copy:
            handler.action_copy_transcription()
            mock_copy.assert_called_once_with("Test transcription")
