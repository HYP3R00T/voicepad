"""Clipboard utilities for VoicePad TUI."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def copy_to_clipboard(text: str) -> None:
    """Copy text to the system clipboard using pyperclip (cross-platform)."""
    try:
        import pyperclip

        pyperclip.copy(text)
    except Exception as e:
        logger.warning(f"Clipboard copy failed: {e}")
