"""Theme and UI constants for the VoicePad TUI."""

from __future__ import annotations


def get_available_themes() -> list[str]:
    """Get list of available Textual themes, sorted alphabetically."""
    return sorted([
        "atom-one-dark",
        "atom-one-light",
        "catppuccin-frappe",
        "catppuccin-latte",
        "catppuccin-macchiato",
        "catppuccin-mocha",
        "dracula",
        "flexoki",
        "gruvbox",
        "monokai",
        "nord",
        "rose-pine",
        "rose-pine-dawn",
        "rose-pine-moon",
        "solarized-dark",
        "solarized-light",
        "textual-dark",
        "textual-light",
        "tokyo-night",
    ])


MD_PLACEHOLDER = """\
# voicepad

Select a recording from the list on the left to view its full transcription here.
"""
