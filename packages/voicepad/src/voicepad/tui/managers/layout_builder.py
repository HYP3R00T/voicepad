"""UI layout composition for VoicePad TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import (
    Footer,
    Label,
    MarkdownViewer,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
)

from voicepad.tui.components.button import VoiceButton

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp

_MD_PLACEHOLDER = """
# No recording selected

Select a recording from the list to view its transcription.
"""


class LayoutBuilder:
    """Builds the UI layout for VoicePadApp."""

    def __init__(self, app: VoicePadApp, version: str) -> None:
        self.app = app
        self.version = version

    def compose(self) -> ComposeResult:
        """Build the complete UI layout."""
        yield from self._compose_header()
        yield Static(id="header-rule")
        yield from self._compose_body()
        yield Footer()

    def _compose_header(self) -> ComposeResult:
        """Build the header section with title, version, status, and model info."""
        with Static(id="header"):
            yield Label("VoicePad", id="header-title")
            yield Label(self.version, id="header-version")
            yield Label("\U000f051f  initialising", id="status")
            yield Label("loading…", id="header-model")

    def _compose_body(self) -> ComposeResult:
        """Build the main body with tabbed content."""
        with Static(id="body"), TabbedContent(id="tabs"):
            yield from self._compose_record_tab()
            yield from self._compose_history_tab()
            yield from self._compose_settings_tab()

    def _compose_record_tab(self) -> ComposeResult:
        """Build the record tab with transcription display."""
        with TabPane("  record  ", id="tab-record"):
            tx = Static(id="transcription")
            tx.border_title = "transcription"
            yield tx

    def _compose_history_tab(self) -> ComposeResult:
        """Build the history tab with list and viewer panes."""
        with TabPane("  history  ", id="tab-history"), Static(id="history-section"):
            hist_list = Static(id="history-list-pane")
            hist_list.border_title = "recordings"
            yield hist_list

            with Static(id="history-view-pane") as view_pane:
                view_pane.border_title = "transcription"
                yield MarkdownViewer(
                    _MD_PLACEHOLDER,
                    id="history-viewer",
                    show_table_of_contents=False,
                    open_links=False,
                )

    def _compose_settings_tab(self) -> ComposeResult:
        """Build the settings tab with form and save button."""
        with TabPane("  settings  ", id="tab-settings"):
            with VerticalScroll(id="settings-scroll"):
                yield Static(id="settings-fields")
            with Static(id="settings-footer"):
                yield Label("", id="settings-status")
                yield VoiceButton("\U000f0493  save", role="primary", id="settings-save-btn")

    def mount_widgets(self) -> None:
        """Mount additional widgets after initial composition."""
        self._mount_record_tab_widgets()
        self._mount_history_tab_widgets()

    def _mount_record_tab_widgets(self) -> None:
        """Mount widgets for the record tab."""
        tx = self.app.query_one("#transcription", Static)
        tx.mount(Label("speak and press space to begin…", id="tx-text", classes="placeholder"))
        tx.mount(Label("", id="tx-meta"))
        tx.mount(VoiceButton("\U000f0191  copy", role="default", id="tx-copy-btn", disabled=True))

    def _mount_history_tab_widgets(self) -> None:
        """Mount widgets for the history tab."""
        hist_list = self.app.query_one("#history-list-pane", Static)
        hist_list.mount(OptionList(id="history-options"))
