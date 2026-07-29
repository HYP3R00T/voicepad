from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _parse_hotkey(hotkey_str: str) -> str | None:
    """Return the hotkey string if valid and non-empty, else None."""
    stripped = hotkey_str.strip()
    return stripped if stripped else None


class GlobalHotkeyListener:
    """Register a system-wide Windows hotkey."""

    def __init__(
        self,
        hotkey: str,
        on_toggle: Callable[[], None],
    ) -> None:
        self._hotkey_str = hotkey
        self._on_toggle = on_toggle
        self._remove_hotkey: Callable[[], None] | None = None

    def start(self) -> None:
        """Register the configured hotkey or raise when registration fails."""
        key = _parse_hotkey(self._hotkey_str)
        if not key:
            logger.info("Global hotkey disabled (empty string)")
            return

        try:
            import keyboard
        except ModuleNotFoundError as error:
            raise RuntimeError("The keyboard package is not installed") from error

        self._remove_hotkey = keyboard.add_hotkey(key, self._on_toggle, suppress=False)
        logger.info("Global hotkey registered: hotkey=%s", key)

    def stop(self) -> None:
        """Unregister the configured hotkey."""
        remove_hotkey = self._remove_hotkey
        if remove_hotkey is None:
            return
        try:
            remove_hotkey()
        except Exception as error:
            logger.warning("Could not unregister global hotkey '%s': %s", self._hotkey_str, error)
        self._remove_hotkey = None
        logger.info("Global hotkey unregistered: hotkey=%s", self._hotkey_str)
