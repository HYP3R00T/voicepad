"""History handler for VoicePad TUI."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
from textual import on, work
from textual.widgets import Button, Markdown, MarkdownViewer, OptionList, TabbedContent
from textual.widgets.option_list import Option

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
                self.app._entries.append(entry)
                self.add_history_entry(entry)

    def add_history_entry(self, entry: SessionEntry) -> None:
        """Add a new entry to the history list."""
        ol = self.app.query_one("#history-options", OptionList)
        name = entry.wav_path.stem if entry.wav_path else f"clip-{entry.index + 1}"
        label = (
            f"[bold]{entry.timestamp}[/]  [dim]{name}[/]\n"
            f"  [dim]{entry.duration_s:.1f}s · {entry.latency_ms:.0f}ms · {entry.device}[/]"
        )
        ol.add_option(Option(label, id=str(entry.index)))
        ol.highlighted = ol.option_count - 1

    @on(OptionList.OptionSelected, "#history-options")
    def on_history_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle selection of a history entry."""
        with contextlib.suppress(Exception):
            idx = int(event.option.id or "-1")
            if 0 <= idx < len(self.app._entries):
                self.app._selected_entry_idx = idx
                entry = self.app._entries[idx]
                self.app.refresh_bindings()
                if entry.md_path and entry.md_path.exists():
                    self.load_history_viewer(entry.md_path)

    @work(name="md-view")
    async def load_history_viewer(self, md_path: Path) -> None:
        """Load and display markdown content in the history viewer."""
        viewer = self.app.query_one("#history-viewer", MarkdownViewer)
        try:
            raw = md_path.read_text(encoding="utf-8")
            lines = raw.splitlines()

            # Parse YAML front matter into per-transcription metadata
            fm_meta: dict[int, dict] = {}
            wav_name = ""
            body_lines: list[str] = lines

            if lines and lines[0].strip() == "---":
                fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
                if fm_end is not None:
                    current: dict | None = None
                    for fl in lines[1:fm_end]:
                        s = fl.strip()
                        if s.startswith("file:"):
                            wav_name = s.split(":", 1)[-1].strip()
                        elif s.startswith("- n:"):
                            with contextlib.suppress(Exception):
                                current = {"n": int(s.split(":")[-1].strip())}
                        elif current is not None and ":" in s:
                            k, _, v = s.partition(":")
                            current[k.strip()] = v.strip()
                            fm_meta[current["n"]] = current
                    body_lines = lines[fm_end + 1 :]

            # Rebuild display content: inject metadata after each ## Transcription N heading
            out: list[str] = []
            if wav_name:
                out += [f"**File:** `{wav_name}`", ""]

            for line in body_lines:
                stripped = line.strip()
                out.append(line)
                if stripped.startswith("## Transcription "):
                    with contextlib.suppress(Exception):
                        n = int(stripped.split()[-1])
                        meta = fm_meta.get(n, {})
                        if meta:
                            parts = []
                            if "model" in meta:
                                parts.append(f"model: {meta['model']}")
                            if "language" in meta:
                                parts.append(f"language: {meta['language']}")
                            if "duration" in meta:
                                parts.append(f"duration: {meta['duration']}")
                            if "latency" in meta:
                                parts.append(f"latency: {meta['latency']}")
                            if "timestamp" in meta:
                                parts.append(f"_{meta['timestamp']}_")
                            if parts:
                                out.append("")
                                out.append("*" + " · ".join(parts) + "*")

            await viewer.document.update("\n".join(out))
        except Exception:
            await viewer.go(md_path.resolve())

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

    @work(thread=True, name="retranscribe")
    def retranscribe_file(self, wav_path: Path, md_path: Path | None) -> None:
        """Retranscribe a WAV file and prepend the result to the markdown."""
        from voicepad_core.transcription import transcribe_buffer

        self.app.call_from_thread(self.app._set_status, "transcribing", f"retranscribing {wav_path.name}…")
        try:
            audio, _sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            result = transcribe_buffer(audio, self.app.config)
            error: str | None = None
        except Exception as e:
            result = None
            error = str(e)

        self.app.call_from_thread(self.on_retranscribe_done, wav_path, md_path, result, error)

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
            btn = self.app.query_one("#tx-copy-btn", Button)
            btn.label = "\U000f012c  copied"
            self.app.set_timer(1.5, lambda: setattr(btn, "label", "\U000f0191  copy"))

    @on(Button.Pressed, "#tx-copy-btn")
    def on_copy_btn_pressed(self) -> None:
        """Handle copy button press."""
        self.action_copy_transcription()
