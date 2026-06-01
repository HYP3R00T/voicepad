"""History service for VoicePad TUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from voicepad.tui.models import SessionEntry
from voicepad.tui.utils.markdown import (
    format_markdown,
    format_markdown_streaming,
    parse_markdown_entry,
    prepend_retranscription,
)

if TYPE_CHECKING:
    from voicepad_core import ChunkResult
    from voicepad_core.config import Config
    from voicepad_core.inference import TranscriptionResult

logger = logging.getLogger(__name__)


class HistoryService:
    """Service for managing transcription history."""

    def __init__(self, config: Config) -> None:
        """Initialize the history service.

        Args:
            config: Application configuration
        """
        self.config = config
        self.entries: list[SessionEntry] = []

    def load_from_disk(self) -> list[SessionEntry]:
        """Load history entries from markdown files on disk.

        Returns:
            List of SessionEntry objects loaded from disk
        """
        self.entries.clear()

        if not self.config.markdown_path.exists():
            logger.info("Markdown directory does not exist, no history to load_model")
            return self.entries

        try:
            md_files = sorted(self.config.markdown_path.glob("*.md"))
            logger.info(f"Found {len(md_files)} markdown files")

            for idx, md_path in enumerate(md_files):
                entry = parse_markdown_entry(md_path, idx, self.config.recordings_path)
                if entry is not None:
                    self.entries.append(entry)
                    logger.debug(f"Loaded entry {idx}: {md_path.name}")

            logger.info(f"Loaded {len(self.entries)} history entries")
        except Exception as e:
            logger.error(f"Failed to load_model history: {e}")

        return self.entries

    def add_entry(self, entry: SessionEntry) -> None:
        """Add a new entry to the history.

        Args:
            entry: The SessionEntry to add
        """
        self.entries.append(entry)
        logger.info(f"Added history entry {entry.index}")

    def get_entry(self, index: int) -> SessionEntry | None:
        """Get an entry by index.

        Args:
            index: The index of the entry to retrieve

        Returns:
            The SessionEntry if found, None otherwise
        """
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def delete_entry(self, index: int) -> bool:
        """Delete an entry and its associated files.

        Args:
            index: The index of the entry to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        if not (0 <= index < len(self.entries)):
            logger.warning(f"Invalid entry index: {index}")
            return False

        entry = self.entries[index]

        # Delete WAV file if it exists
        if entry.wav_path and entry.wav_path.exists():
            try:
                entry.wav_path.unlink()
                logger.info(f"Deleted WAV file: {entry.wav_path}")
            except Exception as e:
                logger.warning(f"Failed to delete WAV file: {e}")

        # Delete markdown file if it exists
        if entry.md_path and entry.md_path.exists():
            try:
                entry.md_path.unlink()
                logger.info(f"Deleted markdown file: {entry.md_path}")
            except Exception as e:
                logger.warning(f"Failed to delete markdown file: {e}")

        # Remove from entries list
        del self.entries[index]

        # Re-index remaining entries
        for new_idx, e in enumerate(self.entries):
            self.entries[new_idx] = SessionEntry(
                index=new_idx,
                wav_path=e.wav_path,
                md_path=e.md_path,
                duration_s=e.duration_s,
                text=e.text,
                latency_ms=e.latency_ms,
                device=e.device,
                timestamp=e.timestamp,
            )

        logger.info(f"Deleted entry {index}, re-indexed {len(self.entries)} remaining entries")
        return True

    def save_markdown(
        self,
        wav_path: Path,
        result: TranscriptionResult,
        model_name: str = "",
    ) -> Path:
        """Save a transcription result to a markdown file.

        Args:
            wav_path: Path to the audio file
            result: The transcription result
            model_name: Name of the model used

        Returns:
            Path to the created markdown file

        Raises:
            Exception: If saving fails
        """
        # Create markdown directory if it doesn't exist
        self.config.markdown_path.mkdir(parents=True, exist_ok=True)

        # Generate markdown content
        md_content = format_markdown(wav_path, result, model_name)

        # Save to file
        md_path = self.config.markdown_path / f"{wav_path.stem}.md"
        try:
            md_path.write_text(md_content, encoding="utf-8")
            logger.info(f"Saved markdown to {md_path}")
            return md_path
        except Exception as e:
            logger.error(f"Failed to save markdown: {e}")
            raise

    def save_markdown_streaming(
        self,
        wav_path: Path,
        text: str,
        duration_s: float,
        chunks: list[ChunkResult],
        model_name: str = "",
    ) -> Path:
        """Save a streaming transcription to a markdown file.

        Args:
            wav_path: Path to the audio file
            text: The transcribed text
            duration_s: Duration of the audio in seconds
            chunks: List of transcription chunks
            model_name: Name of the model used

        Returns:
            Path to the created markdown file

        Raises:
            Exception: If saving fails
        """
        # Create markdown directory if it doesn't exist
        self.config.markdown_path.mkdir(parents=True, exist_ok=True)

        # Generate markdown content
        md_content = format_markdown_streaming(wav_path, text, duration_s, chunks, model_name)

        # Save to file
        md_path = self.config.markdown_path / f"{wav_path.stem}.md"
        try:
            md_path.write_text(md_content, encoding="utf-8")
            logger.info(f"Saved streaming markdown to {md_path}")
            return md_path
        except Exception as e:
            logger.error(f"Failed to save streaming markdown: {e}")
            raise

    def update_markdown_with_retranscription(
        self,
        md_path: Path,
        result: TranscriptionResult,
        model_name: str = "",
    ) -> None:
        """Update a markdown file with a new retranscription.

        Args:
            md_path: Path to the markdown file to update
            result: The new transcription result
            model_name: Name of the model used

        Raises:
            Exception: If updating fails
        """
        try:
            # Prepend the new transcription
            new_content = prepend_retranscription(md_path, result, model_name)

            # Write back to file
            md_path.write_text(new_content, encoding="utf-8")
            logger.info(f"Updated markdown with retranscription: {md_path}")
        except Exception as e:
            logger.error(f"Failed to update markdown with retranscription: {e}")
            raise
