"""Reusable button component for VoicePad TUI.

All interactive buttons in the app use VoiceButton so that hover, focus,
disabled, and colour states are consistent everywhere.

Three visual roles:
  "default"  — dim text, brightens on hover  (Back, Cancel, Close, Copy)
  "primary"  — accent-coloured, bold          (Continue, Download, Save)
  "danger"   — error-coloured, bold           (Delete)
"""

from __future__ import annotations

from textual.widgets import Button


class VoiceButton(Button):
    """A consistently styled, chrome-free button for VoicePad."""

    # Override Textual's built-in Button CSS entirely so no background,
    # border, or padding leaks through regardless of variant.
    DEFAULT_CSS = """
    VoiceButton {
        width: auto;
        height: 1;
        border: none;
        background: transparent;
        padding: 0 1;
        margin: 0;
        min-width: 0;
        text-style: none;
    }
    VoiceButton:hover  { background: transparent; }
    VoiceButton:focus  { background: transparent; }
    VoiceButton.-on    { background: transparent; }

    /* default — secondary / navigation */
    VoiceButton.-role-default          { color: $primary; }
    VoiceButton.-role-default:hover    { color: $foreground; text-style: bold; }
    VoiceButton.-role-default:focus    { color: $foreground; text-style: bold; }

    /* primary — main action */
    VoiceButton.-role-primary          { color: $primary; text-style: bold; }
    VoiceButton.-role-primary:hover    { color: $foreground; }
    VoiceButton.-role-primary:focus    { color: $foreground; }

    /* danger — destructive */
    VoiceButton.-role-danger           { color: $error; text-style: bold; }
    VoiceButton.-role-danger:hover     { color: $foreground; }
    VoiceButton.-role-danger:focus     { color: $foreground; }

    /* disabled — same for all roles */
    VoiceButton:disabled               { color: $text-disabled; text-style: none; }
    """

    def __init__(
        self,
        label: str = "",
        *,
        role: str = "default",
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        name: str | None = None,
    ) -> None:
        # Pass no variant so Textual doesn't apply its own colour rules
        super().__init__(label, id=id, classes=classes, disabled=disabled, name=name)
        self.add_class(f"-role-{role}")
