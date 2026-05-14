"""Tab management and binding control for VoicePad TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import OptionList, TabbedContent

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp


class TabManager:
    """Manages tab switching and context-aware binding visibility."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app

    def on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab activation events."""
        # Auto-select the latest history entry when switching to history tab
        if str(event.tab.id) == "tab-history" and self.app._entries and self.app._selected_entry_idx is None:
            ol = self.app.query_one("#history-options", OptionList)
            if ol.option_count > 0:
                # Only treat _sort_ascending as True when it's an actual bool
                # True; mocks used in tests may return a Mock which should be
                # treated as False here to preserve expected behavior.
                # Select the most recent (last) entry by default so the
                # viewer shows the newest transcription without forcing the
                # list cursor to jump to the bottom.
                target_entry = self.app._entries[-1]

                self.app._selected_entry_idx = target_entry.index
                if target_entry.md_path and target_entry.md_path.exists():
                    self.app._load_history_viewer(target_entry.md_path)
        self.app.refresh_bindings()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Show/enable bindings only for the relevant tab and context."""
        active = self.app.query_one("#tabs", TabbedContent).active if self.app.is_mounted else "tab-record"
        tab_specific: dict[str, str] = {
            "toggle_recording": "tab-record",
            "copy_transcription": "tab-record",
            "retranscribe_entry": "tab-history",
            "delete_entry": "tab-history",
            "toggle_sort_order": "tab-history",
            "save_settings": "tab-settings",
        }
        if action in tab_specific:
            if active != tab_specific[action]:
                return False
            # t and d also require an entry to be selected
            if action in ("retranscribe_entry", "delete_entry"):
                return self.app._selected_entry_idx is not None
        return True
