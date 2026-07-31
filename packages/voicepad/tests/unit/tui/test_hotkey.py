from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from voicepad.tui.hotkey import GlobalHotkeyListener, _parse_hotkey


class TestParseHotkey:
    def test_returns_stripped_string(self) -> None:
        """Whitespace around a configured hotkey is removed."""
        assert _parse_hotkey("  ctrl+alt+v  ") == "ctrl+alt+v"

    def test_returns_none_for_empty_string(self) -> None:
        """An empty hotkey disables registration."""
        assert _parse_hotkey("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        """A whitespace-only hotkey disables registration."""
        assert _parse_hotkey("   ") is None


class TestGlobalHotkeyListener:
    def test_start_registers_callback_and_reports_success(self) -> None:
        """A valid hotkey stores the remover returned after registration."""
        on_toggle = MagicMock()
        remove_hotkey = MagicMock()
        mock_keyboard = MagicMock()
        mock_keyboard.add_hotkey.return_value = remove_hotkey
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_toggle=on_toggle)

        with patch.dict("sys.modules", {"keyboard": mock_keyboard}):
            listener.start()

        mock_keyboard.add_hotkey.assert_called_once_with("ctrl+alt+v", on_toggle, suppress=False)
        assert listener._remove_hotkey is remove_hotkey

    def test_start_does_not_import_keyboard_when_disabled(self) -> None:
        """An empty hotkey returns without importing the platform package."""
        listener = GlobalHotkeyListener(hotkey="", on_toggle=MagicMock())

        with patch("builtins.__import__", side_effect=AssertionError("unexpected import")):
            listener.start()

        assert listener._remove_hotkey is None

    def test_start_raises_clear_error_when_package_is_missing(self) -> None:
        """A missing Windows keyboard dependency produces an actionable error."""
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_toggle=MagicMock())

        with (
            patch("builtins.__import__", side_effect=ModuleNotFoundError("keyboard")),
            pytest.raises(RuntimeError, match="keyboard package is not installed"),
        ):
            listener.start()

    def test_start_propagates_registration_failure(self) -> None:
        """A platform registration error reaches the caller instead of reporting success."""
        mock_keyboard = MagicMock()
        mock_keyboard.add_hotkey.side_effect = PermissionError("denied")
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_toggle=MagicMock())

        with (
            patch.dict("sys.modules", {"keyboard": mock_keyboard}),
            pytest.raises(PermissionError, match="denied"),
        ):
            listener.start()

    def test_stop_calls_registered_remover(self) -> None:
        """Stopping an active listener invokes the exact remover returned at registration."""
        remove_hotkey = MagicMock()
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_toggle=MagicMock())
        listener._remove_hotkey = remove_hotkey

        listener.stop()

        remove_hotkey.assert_called_once_with()

    def test_stop_is_idempotent(self) -> None:
        """Stopping an inactive listener performs no work."""
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_toggle=MagicMock())

        listener.stop()

        assert listener._remove_hotkey is None

    def test_stop_clears_remover_after_failure(self) -> None:
        """A failed unregistration is logged and does not leave stale active state."""
        listener = GlobalHotkeyListener(hotkey="ctrl+alt+v", on_toggle=MagicMock())
        listener._remove_hotkey = MagicMock(side_effect=RuntimeError("remove failed"))

        listener.stop()

        assert listener._remove_hotkey is None
