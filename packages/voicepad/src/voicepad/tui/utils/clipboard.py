"""Clipboard utilities for VoicePad TUI."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def copy_to_clipboard(text: str) -> bool:
    """Copy text and report whether the desktop clipboard accepted it."""
    try:
        import pyperclip

        pyperclip.copy(text)
    except Exception as error:
        logger.warning("Clipboard copy failed: %s", error)
        return False
    return True
