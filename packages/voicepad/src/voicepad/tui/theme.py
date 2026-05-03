"""Theme and UI constants for the VoicePad TUI."""

from __future__ import annotations

from textual.theme import Theme

THEME_NAME = "voicepad-dark"

CATPPUCCIN_MOCHA_BLUE = Theme(
    name=THEME_NAME,
    primary="#89b4fa",
    secondary="#74c7ec",
    warning="#FAE3B0",
    error="#F28FAD",
    success="#ABE9B3",
    accent="#fab387",
    foreground="#cdd6f4",
    background="#181825",
    surface="#313244",
    panel="#45475a",
    variables={
        "input-cursor-foreground": "#11111b",
        "input-cursor-background": "#f5e0dc",
        "input-selection-background": "#9399b2 30%",
        "border": "#89b4fa",
        "border-blurred": "#585b70",
        "footer-background": "#45475a",
        "footer-key-foreground": "#89b4fa",
        "block-cursor-foreground": "#1e1e2e",
        "block-cursor-text-style": "none",
        "button-color-foreground": "#181825",
    },
)


def get_available_themes() -> list[str]:
    """Get list of available themes including built-in Textual themes and custom themes."""
    # Built-in Textual themes
    builtin_themes = [
        "textual-dark",
        "textual-light",
        "nord",
        "gruvbox",
        "catppuccin-mocha",
        "dracula",
        "tokyo-night",
        "monokai",
        "flexoki",
        "catppuccin-latte",
        "catppuccin-frappe",
        "catppuccin-macchiato",
        "solarized-light",
        "solarized-dark",
        "rose-pine",
        "rose-pine-moon",
        "rose-pine-dawn",
        "atom-one-dark",
        "atom-one-light",
    ]

    # Custom themes
    custom_themes = [
        "voicepad-dark",  # VoicePad's custom dark theme with blue accents
    ]

    return custom_themes + builtin_themes


MD_PLACEHOLDER = """\
# voicepad

Select a recording from the list on the left to view its full transcription here.
"""
