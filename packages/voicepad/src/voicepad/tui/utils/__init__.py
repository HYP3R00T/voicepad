"""Utility modules for VoicePad TUI."""

from voicepad.tui.utils.clipboard import copy_to_clipboard
from voicepad.tui.utils.hotkey_utils import (
    HOTKEY_KEYS,
    MOD_TO_DISPLAY,
    MOD_TO_KEYBOARD,
    build_hotkey_display_str,
    build_hotkey_str,
    parse_hotkey_str,
)
from voicepad.tui.utils.markdown import (
    format_markdown,
    format_markdown_streaming,
    parse_markdown_entry,
    prepend_retranscription,
)
from voicepad.tui.utils.timer import RecordingTimer

__all__ = [
    "copy_to_clipboard",
    "RecordingTimer",
    "HOTKEY_KEYS",
    "MOD_TO_KEYBOARD",
    "MOD_TO_DISPLAY",
    "build_hotkey_str",
    "build_hotkey_display_str",
    "parse_hotkey_str",
    "format_markdown",
    "format_markdown_streaming",
    "parse_markdown_entry",
    "prepend_retranscription",
]
