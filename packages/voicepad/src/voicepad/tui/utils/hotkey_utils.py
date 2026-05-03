"""Hotkey string parsing and building utilities."""

from __future__ import annotations

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

MOD_TO_PYNPUT: dict[str, str] = {
    "ctrl": "<ctrl>",
    "alt": "<alt>",
    "shift": "<shift>",
    "cmd": "<cmd>",  # Windows key on Windows, Command key on macOS
}


def parse_hotkey_str(hotkey: str) -> tuple[list[str], str]:
    """Parse '<ctrl>+<alt>+v' → (['ctrl', 'alt'], 'v')."""
    mods: list[str] = []
    key = "v"
    for part in hotkey.lower().split("+"):
        part = part.strip().strip("<>")
        if part in MOD_TO_PYNPUT:
            mods.append(part)
        elif part:
            key = part
    return mods, key


def build_hotkey_str(mods: list[str], key: str) -> str:
    """Build '<ctrl>+<alt>+v' from (['ctrl', 'alt'], 'v')."""
    if not key:
        return ""
    parts = [MOD_TO_PYNPUT[m] for m in mods if m in MOD_TO_PYNPUT]
    key_part = key if len(key) == 1 else f"<{key}>"
    parts.append(key_part)
    return "+".join(parts)
