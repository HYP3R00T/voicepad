"""Backward-compatibility shim — re-exports from voicepad.tui.utils.markdown.

Tests and other code that import from voicepad.tui.markdown will continue
to work after the utils subpackage reorganisation.
"""

from voicepad.tui.utils.markdown import (
    format_markdown,
    format_markdown_streaming,
    parse_markdown_entry,
    prepend_retranscription,
)

__all__ = [
    "format_markdown",
    "format_markdown_streaming",
    "parse_markdown_entry",
    "prepend_retranscription",
]
