"""Hotkey string parsing and building utilities."""

from __future__ import annotations

import platform

# Keys available in the hotkey picker
HOTKEY_KEYS: list[str] = [
    *"abcdefghijklmnopqrstuvwxyz",
    *"0123456789",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "space",
    "tab",
    "enter",
    "backspace",
    "delete",
    "insert",
    "home",
    "end",
    "page_up",
    "page_down",
    "up",
    "down",
    "left",
    "right",
]

# Mapping for keyboard module format (no angle brackets)
MOD_TO_KEYBOARD: dict[str, str] = {
    "ctrl": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "cmd": "win",  # Windows key on Windows, Command key on macOS
}

# Mapping for display purposes
MOD_TO_DISPLAY: dict[str, str] = {
    "ctrl": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "cmd": "win",  # Display as "win" for Windows key
}

_SUPER_ALIASES = {"cmd", "super", "win", "windows"}


def parse_hotkey_str(hotkey: str) -> tuple[list[str], str]:
    """Parse 'ctrl+alt+v' → (['ctrl', 'alt'], 'v').

    Handles both old format with angle brackets and new format without.
    """
    mods: list[str] = []
    key = "v"
    for part in hotkey.lower().split("+"):
        part = part.strip().strip("<>")
        if part in _SUPER_ALIASES:
            mods.append("cmd")
        elif part in MOD_TO_KEYBOARD:
            mods.append(part)
        elif part:
            key = part
    return mods, key


def build_hotkey_str(mods: list[str], key: str) -> str:
    """Build 'ctrl+alt+v' from (['ctrl', 'alt'], 'v') for keyboard module.

    The keyboard module expects lowercase format without angle brackets:
    - "ctrl+shift+space" (correct)
    - "<ctrl>+<shift>+<space>" (incorrect)
    """
    if not key:
        return ""
    parts = [MOD_TO_KEYBOARD[m] for m in mods if m in MOD_TO_KEYBOARD]
    parts.append(key)
    return "+".join(parts)


def build_hotkey_display_str(mods: list[str], key: str) -> str:
    """Build 'ctrl+alt+v' from (['ctrl', 'alt'], 'v') for display purposes.

    Uses 'win' instead of 'cmd' for better clarity on Windows systems.
    Same format as build_hotkey_str since keyboard module uses simple format.
    """
    if not key:
        return ""
    display_mapping = dict(MOD_TO_DISPLAY)
    if platform.system() != "Windows":
        display_mapping["cmd"] = "super"
    parts = [display_mapping[m] for m in mods if m in display_mapping]
    parts.append(key)
    return "+".join(parts)
