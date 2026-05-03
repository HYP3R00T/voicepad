"""Minimal checkbox component for VoicePad TUI.

A chrome-free checkbox that matches VoiceButton's minimal aesthetic.
Shows checked/unchecked state with simple text indicators, no borders or backgrounds.
"""

from __future__ import annotations

import logging

from textual.widgets import Checkbox

logger = logging.getLogger(__name__)


class VoiceCheckbox(Checkbox):
    """A minimal, chrome-free checkbox for VoicePad."""

    # Custom button symbols - MUST be class attributes for Textual to use them
    BUTTON_INNER = "◯"  # Unchecked: large empty circle (ring) - U+25EF
    BUTTON_CHECKED = "⬤"  # Checked: large filled circle (disc) - U+2B24

    # Override Textual's built-in Checkbox CSS to remove all chrome
    DEFAULT_CSS = """
    VoiceCheckbox {
        width: auto;
        height: 1;
        border: none !important;
        background: transparent !important;
        padding: 0;
        margin: 0;
        min-width: 0;
        outline: none !important;
    }

    VoiceCheckbox:hover {
        background: transparent !important;
        border: none !important;
        outline: none !important;
    }

    VoiceCheckbox:focus {
        background: transparent !important;
        border: none !important;
        outline: none !important;
    }

    /* Unchecked state - RED to indicate OFF */
    VoiceCheckbox > .toggle--button {
        border: none !important;
        background: transparent !important;
        color: $error;
        width: 4;
        padding: 0;
        outline: none !important;
    }

    /* Checked state - GREEN to indicate ON */
    VoiceCheckbox.-on > .toggle--button {
        background: transparent !important;
        border: none !important;
        color: $success;
        text-style: bold;
        outline: none !important;
    }

    /* Hover states - brighten */
    VoiceCheckbox:hover > .toggle--button {
        background: transparent !important;
        border: none !important;
        color: $error 80%;
        outline: none !important;
    }

    VoiceCheckbox.-on:hover > .toggle--button {
        background: transparent !important;
        border: none !important;
        color: $success 80%;
        text-style: bold;
        outline: none !important;
    }

    /* Focus states - no visual change, just brighten color */
    VoiceCheckbox:focus > .toggle--button {
        background: transparent !important;
        border: none !important;
        color: $error 80%;
        outline: none !important;
    }

    VoiceCheckbox.-on:focus > .toggle--button {
        background: transparent !important;
        border: none !important;
        color: $success 80%;
        text-style: bold;
        outline: none !important;
    }

    /* Label styling */
    VoiceCheckbox > .toggle--label {
        color: $text;
        padding: 0 0 0 1;
        background: transparent !important;
        border: none !important;
    }

    VoiceCheckbox:hover > .toggle--label {
        color: $foreground;
        background: transparent !important;
        border: none !important;
    }

    VoiceCheckbox:focus > .toggle--label {
        color: $foreground;
        background: transparent !important;
        border: none !important;
    }

    /* Disabled state */
    VoiceCheckbox:disabled > .toggle--button {
        color: $text-disabled;
        background: transparent !important;
        border: none !important;
    }

    VoiceCheckbox:disabled > .toggle--label {
        color: $text-disabled;
        background: transparent !important;
        border: none !important;
    }
    """

    def __init__(
        self,
        label: str = "",
        *,
        value: bool = False,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize a minimal checkbox.

        Args:
            label: The text label for the checkbox
            value: Initial checked state
            id: Widget ID
            classes: CSS classes
            disabled: Whether the checkbox is disabled
            name: Widget name
        """
        super().__init__(
            label,
            value=value,
            button_first=True,
            id=id,
            classes=classes,
            disabled=disabled,
            name=name,
        )
        logger.debug(f"VoiceCheckbox created: {self.id}, value={self.value}")
