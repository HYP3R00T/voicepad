"""Global hotkey listener for VoicePad.

Runs in a background thread using pynput. When the configured hotkey is
pressed from any application:
  - First press  → calls on_start()
  - Second press → calls on_stop()

The caller is responsible for transcription and clipboard copy after on_stop().
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _parse_hotkey(hotkey_str: str) -> str | None:
    """Return the hotkey string if valid and non-empty, else None."""
    stripped = hotkey_str.strip()
    return stripped if stripped else None


class GlobalHotkeyListener:
    """Listens for a system-wide hotkey and toggles recording state.

    Usage:
        listener = GlobalHotkeyListener(
            hotkey="<ctrl>+<alt>+v",
            on_start=lambda: ...,
            on_stop=lambda: ...,
        )
        listener.start()
        # ... app runs ...
        listener.stop()
    """

    def __init__(
        self,
        hotkey: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self._hotkey_str = hotkey
        self._on_start = on_start
        self._on_stop = on_stop
        self._recording = False
        self._listener: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background hotkey listener thread."""
        key = _parse_hotkey(self._hotkey_str)
        if not key:
            logger.info("Global hotkey disabled (empty string)")
            return

        self._thread = threading.Thread(target=self._run, daemon=True, name="hotkey-listener")
        self._thread.start()
        logger.info(f"Global hotkey listener started: {key}")

    def stop(self) -> None:
        """Stop the listener."""
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._listener.stop()
        logger.info("Global hotkey listener stopped")

    def _run(self) -> None:
        """Background thread: block on pynput hotkey listener."""
        key = _parse_hotkey(self._hotkey_str)
        if not key:
            return

        try:
            from pynput import keyboard

            def _on_activate() -> None:
                if self._recording:
                    self._recording = False
                    logger.debug("Global hotkey: stop recording")
                    try:
                        self._on_stop()
                    except Exception as e:
                        logger.error(f"Hotkey on_stop error: {e}")
                else:
                    self._recording = True
                    logger.debug("Global hotkey: start recording")
                    try:
                        self._on_start()
                    except Exception as e:
                        logger.error(f"Hotkey on_start error: {e}")

            hotkey_map = {key: _on_activate}
            self._listener = keyboard.GlobalHotKeys(hotkey_map)
            self._listener.run()  # type: ignore[union-attr]

        except Exception as e:
            logger.error(f"Global hotkey listener failed: {e}")
