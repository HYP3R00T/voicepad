"""History handler for VoicePad TUI."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from textual import on
from textual.widgets import Button, Markdown, MarkdownViewer, OptionList, TabbedContent
from textual.widgets.option_list import Option

from voicepad.tui.components import VoiceButton
from voicepad.tui.modals import DeleteConfirmModal
from voicepad.tui.models import SessionEntry
from voicepad.tui.theme import MD_PLACEHOLDER as _MD_PLACEHOLDER
from voicepad.tui.utils.clipboard import copy_to_clipboard as _copy_to_clipboard
from voicepad.tui.utils.markdown import parse_markdown_entry as _parse_markdown_entry
from voicepad.tui.utils.markdown import prepend_retranscription as _prepend_retranscription

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp


class HistoryHandler:
    """Handles history tab functionality."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app

    def load_history_from_disk(self) -> None:
        """Pre-populate history from existing markdown files."""
        md_dir = self.app.config.markdown_path
        if not md_dir.exists():
            return
        for md_path in sorted(md_dir.glob("*.md")):
            entry = _parse_markdown_entry(
                md_path, index=len(self.app._entries), recordings_path=self.app.config.recordings_path
            )
            if entry is not None:
                # Append entry to the canonical chronological list
                self.app._entries.append(entry)
        # Refresh the displayed list once after all entries are loaded
        self.refresh_history_list()

    def add_history_entry(self, entry: SessionEntry) -> None:
        """Add a new entry to the history list and refresh the display.

        Note: The caller is responsible for appending the entry to self.app._entries.
        This method only refreshes the displayed OptionList.
        """
        # Ensure there's a list to work with (tests may provide a Mock)
        if not hasattr(self.app, "_entries") or not isinstance(self.app._entries, list):
            self.app._entries = []

        # Refresh the displayed OptionList so it respects the current sort order
        self.refresh_history_list()

    def on_history_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle selection of a history entry."""
        try:
            idx = int(event.option.id or "-1")
            if 0 <= idx < len(self.app._entries):
                self.app._selected_entry_idx = idx
                entry = self.app._entries[idx]
                self.app.refresh_bindings()
                if entry.md_path and entry.md_path.exists():
                    self.load_history_viewer(entry.md_path)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to select history entry: {e}")

    def load_history_viewer(self, md_path: Path) -> None:
        """Load and display markdown content in the history viewer."""
        # Delegate to app's worker method (App is a DOMNode, so @work decorator works there)
        self.app._load_history_viewer(md_path)

    @on(Markdown.LinkClicked)
    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks in the markdown viewer by opening them in the system browser."""
        import webbrowser

        try:
            webbrowser.open(event.href)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to open link {event.href}: {e}")

    def action_retranscribe_entry(self) -> None:
        """Retranscribe the selected history entry via keyboard shortcut."""
        if self.app._selected_entry_idx is None or not self.app._model_ready:
            return
        entry = self.app._entries[self.app._selected_entry_idx]
        if entry.wav_path and entry.wav_path.exists():
            self.retranscribe_file(entry.wav_path, entry.md_path)

    def retranscribe_file(self, wav_path: Path, md_path: Path | None) -> None:
        """Retranscribe a WAV file and prepend the result to the markdown."""
        # Delegate to app's worker method (App is a DOMNode, so @work decorator works there)
        self.app._retranscribe_file(wav_path, md_path)

    def on_retranscribe_done(self, wav_path: Path, md_path: Path | None, result, error: str | None) -> None:
        """Handle completion of retranscription."""
        if error:
            self.app._set_status("error", error)
            return

        if result:
            # Prepend new transcription — never overwrite the existing ones
            out_md = md_path or (self.app.config.markdown_path / f"{wav_path.stem}.md")
            self.app.config.markdown_path.mkdir(parents=True, exist_ok=True)
            new_content = _prepend_retranscription(out_md, result, self.app.config.transcription_model)
            out_md.write_text(new_content, encoding="utf-8")
            self.app._set_status("ready", "ready")

            # Update the in-memory entry if it exists
            if self.app._selected_entry_idx is not None:
                entry = self.app._entries[self.app._selected_entry_idx]
                self.app._entries[self.app._selected_entry_idx] = SessionEntry(
                    index=entry.index,
                    wav_path=entry.wav_path,
                    md_path=out_md,
                    duration_s=result.duration_s,
                    text=result.text,
                    latency_ms=result.latency_ms,
                    device=result.device,
                    timestamp=entry.timestamp,
                )

            # Reload the viewer with the fresh markdown
            self.load_history_viewer(out_md)

    def action_delete_entry(self) -> None:
        """Show delete confirmation for the selected history entry."""
        active = self.app.query_one("#tabs", TabbedContent).active
        if active != "tab-history" or self.app._selected_entry_idx is None:
            return
        self.show_delete_confirm()

    def toggle_sort_order(self) -> None:
        """Toggle between ascending and descending sort order for history entries."""
        self.app._sort_ascending = not self.app._sort_ascending
        self.refresh_history_list()

        # Show a transient sort notification only for explicit user toggles.
        sort_status = "↓" if self.app._sort_ascending else "↑"
        self.app._set_status("ready", f"Sorted {sort_status}")
        self.app.set_timer(1.5, lambda: self.app._set_status("ready", "ready"))

    def refresh_history_list(self) -> None:
        """Refresh the history list by clearing and re-adding entries in sorted order."""
        import logging

        logger = logging.getLogger(__name__)

        if not self.app._entries:
            return

        # Sort entries by index (chronological order). Only treat
        # `_sort_ascending` as True when it's explicitly a bool True; tests
        # may provide a Mock object instead.
        sort_newest_first = getattr(self.app, "_sort_ascending", False)
        if not isinstance(sort_newest_first, bool):
            sort_newest_first = False

        # When `sort_newest_first` is True we want newest-first display,
        # which means reversing the chronological order so highest index
        # appears first.
        sorted_entries = sorted(self.app._entries, key=lambda e: e.index, reverse=sort_newest_first)

        # Clear and rebuild the OptionList
        ol = self.app.query_one("#history-options", OptionList)
        ol.clear_options()

        for e in sorted_entries:
            name = e.wav_path.stem if e.wav_path else f"clip-{e.index + 1}"
            label = (
                f"[bold]{e.timestamp}[/]  [dim]{name}[/]\n"
                f"  [dim]{e.duration_s:.1f}s · {e.latency_ms:.0f}ms · {e.device}[/]"
            )
            ol.add_option(Option(label, id=str(e.index)))
        # Determine which entry should be highlighted (and therefore shown
        # at the top of the OptionList viewport). When the UI is showing
        # newest-first, highlight the first option; otherwise highlight the
        # last option.
        if sort_newest_first:
            ol.highlighted = 0
            # The newest entry is the first in sorted_entries when newest-first
            self.app._selected_entry_idx = sorted_entries[0].index
        else:
            ol.highlighted = ol.option_count - 1
            self.app._selected_entry_idx = sorted_entries[-1].index

        logger.info(
            f"History list refreshed with sort order: {'newest first' if self.app._sort_ascending else 'oldest first'}"
        )

    def show_delete_confirm(self) -> None:
        """Show the delete confirmation modal."""
        if self.app._selected_entry_idx is None:
            return
        entry = self.app._entries[self.app._selected_entry_idx]
        name = entry.wav_path.stem if entry.wav_path else f"clip-{entry.index + 1}"

        def _delete_callback(result: bool | None) -> None:
            if result is not None:
                self.on_delete_confirmed(result)

        self.app.push_screen(DeleteConfirmModal(name), callback=_delete_callback)

    def on_delete_confirmed(self, confirmed: bool) -> None:
        """Handle confirmation of entry deletion."""
        if not confirmed or self.app._selected_entry_idx is None:
            return
        entry = self.app._entries[self.app._selected_entry_idx]

        # Delete files from disk
        with contextlib.suppress(Exception):
            if entry.wav_path and entry.wav_path.exists():
                entry.wav_path.unlink()
        with contextlib.suppress(Exception):
            if entry.md_path and entry.md_path.exists():
                entry.md_path.unlink()

        # Remove from in-memory list
        del self.app._entries[self.app._selected_entry_idx]
        self.app._selected_entry_idx = None

        # Rebuild the OptionList from scratch
        ol = self.app.query_one("#history-options", OptionList)
        ol.clear_options()
        for new_idx, e in enumerate(self.app._entries):
            e = SessionEntry(
                index=new_idx,
                wav_path=e.wav_path,
                md_path=e.md_path,
                duration_s=e.duration_s,
                text=e.text,
                latency_ms=e.latency_ms,
                device=e.device,
                timestamp=e.timestamp,
            )
            self.app._entries[new_idx] = e
            name = e.wav_path.stem if e.wav_path else f"clip-{new_idx + 1}"
            label = (
                f"[bold]{e.timestamp}[/]  [dim]{name}[/]\n"
                f"  [dim]{e.duration_s:.1f}s · {e.latency_ms:.0f}ms · {e.device}[/]"
            )
            ol.add_option(Option(label, id=str(new_idx)))

        # Clear the viewer and disable action buttons
        self.app.refresh_bindings()
        with contextlib.suppress(Exception):
            self.app.run_worker(
                self.app.query_one("#history-viewer", MarkdownViewer).document.update(_MD_PLACEHOLDER),
                name="md-clear",
            )

    def action_copy_transcription(self) -> None:
        """Copy the current transcription to clipboard."""
        if not self.app._current_text:
            return
        _copy_to_clipboard(self.app._current_text)
        with contextlib.suppress(Exception):
            btn = self.app.query_one("#tx-copy-btn", VoiceButton)
            btn.label = "\U000f012c  copied"
            self.app.set_timer(1.5, lambda: setattr(btn, "label", "\U000f0191  copy"))

    @on(Button.Pressed, "#tx-copy-btn")
    def on_copy_btn_pressed(self) -> None:
        """Handle copy button press."""
        self.action_copy_transcription()
