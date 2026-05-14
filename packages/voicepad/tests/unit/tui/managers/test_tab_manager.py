"""Tests for TabManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.widgets import TabbedContent


class TestTabManagerInit:
    """Tests for TabManager initialization."""

    def test_init_stores_app_reference(self) -> None:
        """Test that __init__ stores the app reference."""
        from voicepad.tui.managers.tab_manager import TabManager

        mock_app = Mock()
        manager = TabManager(mock_app)

        assert manager.app is mock_app


class TestOnTabActivated:
    """Tests for on_tab_activated method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app._entries = []
        app._selected_entry_idx = None
        app.refresh_bindings = Mock()
        return app

    def test_refreshes_bindings_on_any_tab(self, mock_app: Mock) -> None:
        """Test that refresh_bindings is called on any tab activation."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)
        event = Mock()
        event.tab.id = "tab-record"

        manager.on_tab_activated(event)

        mock_app.refresh_bindings.assert_called_once()

    def test_auto_selects_latest_entry_on_history_tab(self, mock_app: Mock) -> None:
        """Test that switching to history tab auto-selects the latest entry."""
        from voicepad.tui.managers.tab_manager import TabManager

        # Create mock entries
        entry1 = Mock()
        entry1.index = 1
        entry1.md_path = Mock(spec=Path)
        entry1.md_path.exists.return_value = True

        entry2 = Mock()
        entry2.index = 2
        entry2.md_path = Mock(spec=Path)
        entry2.md_path.exists.return_value = True

        mock_app._entries = [entry1, entry2]
        mock_app._selected_entry_idx = None
        mock_app._load_history_viewer = Mock()

        # Mock OptionList
        mock_ol = Mock()
        mock_ol.option_count = 2
        mock_app.query_one = Mock(return_value=mock_ol)

        manager = TabManager(mock_app)
        event = Mock()
        event.tab.id = "tab-history"

        manager.on_tab_activated(event)

        # Should select last entry
        assert mock_app._selected_entry_idx == 2
        # Should load the history viewer
        mock_app._load_history_viewer.assert_called_once_with(entry2.md_path)

    def test_does_not_force_history_cursor_to_bottom(self, mock_app: Mock) -> None:
        """Test that history activation does not force the list cursor to the end."""
        from voicepad.tui.managers.tab_manager import TabManager

        entry = Mock()
        entry.index = 1
        entry.md_path = Mock(spec=Path)
        entry.md_path.exists.return_value = True

        mock_app._entries = [entry]
        mock_app._selected_entry_idx = None
        mock_app._load_history_viewer = Mock()

        mock_ol = Mock()
        mock_ol.option_count = 1
        mock_app.query_one = Mock(return_value=mock_ol)

        manager = TabManager(mock_app)
        event = Mock()
        event.tab.id = "tab-history"

        manager.on_tab_activated(event)

        assert mock_app._selected_entry_idx == 1
        mock_app._load_history_viewer.assert_called_once_with(entry.md_path)
        # The tab activation should not force a cursor position on the list.
        assert mock_ol.highlighted.call_count == 0

    def test_does_not_auto_select_if_already_selected(self, mock_app: Mock) -> None:
        """Test that auto-selection is skipped if an entry is already selected."""
        from voicepad.tui.managers.tab_manager import TabManager

        entry = Mock()
        entry.index = 1
        entry.md_path = Path("/tmp/entry.md")

        mock_app._entries = [entry]
        mock_app._selected_entry_idx = 1  # Already selected
        mock_app._load_history_viewer = Mock()

        manager = TabManager(mock_app)
        event = Mock()
        event.tab.id = "tab-history"

        manager.on_tab_activated(event)

        # Should not call query_one or load_history_viewer
        mock_app.query_one.assert_not_called()
        mock_app._load_history_viewer.assert_not_called()

    def test_does_not_auto_select_if_no_entries(self, mock_app: Mock) -> None:
        """Test that auto-selection is skipped if there are no entries."""
        from voicepad.tui.managers.tab_manager import TabManager

        mock_app._entries = []
        mock_app._selected_entry_idx = None

        manager = TabManager(mock_app)
        event = Mock()
        event.tab.id = "tab-history"

        manager.on_tab_activated(event)

        # Should not call query_one
        mock_app.query_one.assert_not_called()

    def test_does_not_load_viewer_if_md_path_missing(self, mock_app: Mock) -> None:
        """Test that viewer is not loaded if md_path is None."""
        from voicepad.tui.managers.tab_manager import TabManager

        entry = Mock()
        entry.index = 1
        entry.md_path = None

        mock_app._entries = [entry]
        mock_app._selected_entry_idx = None
        mock_app._load_history_viewer = Mock()

        mock_ol = Mock()
        mock_ol.option_count = 1
        mock_app.query_one = Mock(return_value=mock_ol)

        manager = TabManager(mock_app)
        event = Mock()
        event.tab.id = "tab-history"

        manager.on_tab_activated(event)

        # Should select entry but not load viewer
        assert mock_app._selected_entry_idx == 1
        mock_app._load_history_viewer.assert_not_called()

    def test_does_not_load_viewer_if_md_path_not_exists(self, mock_app: Mock) -> None:
        """Test that viewer is not loaded if md_path does not exist."""
        from voicepad.tui.managers.tab_manager import TabManager

        entry = Mock()
        entry.index = 1
        entry.md_path = Mock(spec=Path)
        entry.md_path.exists.return_value = False

        mock_app._entries = [entry]
        mock_app._selected_entry_idx = None
        mock_app._load_history_viewer = Mock()

        mock_ol = Mock()
        mock_ol.option_count = 1
        mock_app.query_one = Mock(return_value=mock_ol)

        manager = TabManager(mock_app)
        event = Mock()
        event.tab.id = "tab-history"

        manager.on_tab_activated(event)

        # Should select entry but not load viewer
        assert mock_app._selected_entry_idx == 1
        mock_app._load_history_viewer.assert_not_called()


class TestCheckAction:
    """Tests for check_action method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app.is_mounted = True
        app._selected_entry_idx = None

        # Mock TabbedContent
        mock_tabs = Mock(spec=TabbedContent)
        mock_tabs.active = "tab-record"
        app.query_one = Mock(return_value=mock_tabs)

        return app

    def test_allows_action_on_correct_tab(self, mock_app: Mock) -> None:
        """Test that actions are allowed on their designated tab."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        # Set active tab to record
        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-record"

        # toggle_recording should be allowed on tab-record
        result = manager.check_action("toggle_recording", ())
        assert result is True

    def test_blocks_action_on_wrong_tab(self, mock_app: Mock) -> None:
        """Test that actions are blocked on wrong tabs."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        # Set active tab to settings
        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-settings"

        # toggle_recording should be blocked on tab-settings
        result = manager.check_action("toggle_recording", ())
        assert result is False

    def test_allows_non_tab_specific_actions(self, mock_app: Mock) -> None:
        """Test that non-tab-specific actions are always allowed."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        # quit is not tab-specific
        result = manager.check_action("quit", ())
        assert result is True

    def test_blocks_retranscribe_without_selection(self, mock_app: Mock) -> None:
        """Test that retranscribe_entry requires a selected entry."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        # Set active tab to history
        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-history"
        mock_app._selected_entry_idx = None

        # retranscribe_entry should be blocked without selection
        result = manager.check_action("retranscribe_entry", ())
        assert result is False

    def test_blocks_open_recording_without_selection(self, mock_app: Mock) -> None:
        """Test that open_recording requires a selected entry."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-history"
        mock_app._selected_entry_idx = None

        result = manager.check_action("open_recording", ())
        assert result is False

    def test_blocks_open_markdown_without_selection(self, mock_app: Mock) -> None:
        """Test that open_markdown requires a selected entry."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-history"
        mock_app._selected_entry_idx = None

        result = manager.check_action("open_markdown", ())
        assert result is False

    def test_allows_retranscribe_with_selection(self, mock_app: Mock) -> None:
        """Test that retranscribe_entry is allowed with a selected entry."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        # Set active tab to history
        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-history"
        mock_app._selected_entry_idx = 1

        # retranscribe_entry should be allowed with selection
        result = manager.check_action("retranscribe_entry", ())
        assert result is True

    def test_blocks_delete_without_selection(self, mock_app: Mock) -> None:
        """Test that delete_entry requires a selected entry."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        # Set active tab to history
        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-history"
        mock_app._selected_entry_idx = None

        # delete_entry should be blocked without selection
        result = manager.check_action("delete_entry", ())
        assert result is False

    def test_allows_delete_with_selection(self, mock_app: Mock) -> None:
        """Test that delete_entry is allowed with a selected entry."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)

        # Set active tab to history
        mock_tabs = mock_app.query_one.return_value
        mock_tabs.active = "tab-history"
        mock_app._selected_entry_idx = 1

        # delete_entry should be allowed with selection
        result = manager.check_action("delete_entry", ())
        assert result is True

    def test_handles_unmounted_app(self, mock_app: Mock) -> None:
        """Test that check_action handles unmounted app gracefully."""
        from voicepad.tui.managers.tab_manager import TabManager

        mock_app.is_mounted = False
        manager = TabManager(mock_app)

        # Should default to tab-record when not mounted
        result = manager.check_action("toggle_recording", ())
        assert result is True

    def test_all_tab_specific_actions(self, mock_app: Mock) -> None:
        """Test all tab-specific actions are correctly mapped."""
        from voicepad.tui.managers.tab_manager import TabManager

        manager = TabManager(mock_app)
        mock_tabs = mock_app.query_one.return_value

        # Test each tab-specific action
        test_cases = [
            ("toggle_recording", "tab-record", True),
            ("copy_transcription", "tab-record", True),
            ("retranscribe_entry", "tab-history", False),  # Needs selection
            ("delete_entry", "tab-history", False),  # Needs selection
            ("save_settings", "tab-settings", True),
        ]

        for action, tab, expected in test_cases:
            mock_tabs.active = tab
            result = manager.check_action(action, ())
            assert result is expected, f"Action {action} on {tab} should return {expected}"
