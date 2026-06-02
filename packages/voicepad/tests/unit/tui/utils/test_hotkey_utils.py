from __future__ import annotations

from voicepad.tui.utils.hotkey_utils import build_hotkey_str, parse_hotkey_str


class TestHotkeyUtils:
    def test_parse_with_mods_and_key(self) -> None:
        """Parses a modifier+key string into modifiers and key."""
        assert parse_hotkey_str("<ctrl>+<alt>+V") == (["ctrl", "alt"], "v")

    def test_parse_without_mods_returns_key(self) -> None:
        """Parses a plain key string and returns no modifiers."""
        assert parse_hotkey_str("x") == ([], "x")

    def test_build_with_empty_key_returns_empty(self) -> None:
        """Building a hotkey with an empty key returns an empty string."""
        assert build_hotkey_str([], "") == ""

    def test_build_wraps_multi_char_key(self) -> None:
        """Multi-character keys are built in keyboard module format (no angle brackets)."""
        assert build_hotkey_str(["ctrl"], "space") == "ctrl+space"
