"""Global hotkey listener for VoicePad.

Runs in a background thread using the keyboard library. When the configured
hotkey is pressed from any application:
  - First press  → calls on_start()
  - Second press → calls on_stop()

The caller is responsible for transcription and clipboard copy after on_stop().
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _parse_hotkey(hotkey_str: str) -> str | None:
    """Return the hotkey string if valid and non-empty, else None."""
    stripped = hotkey_str.strip()
    return stripped if stripped else None


class GlobalHotkeyListener:
    """Listens for a system-wide hotkey and toggles recording state.

    Uses the keyboard library for global hotkey detection on Windows.
    Supports natural hotkey strings like "ctrl+shift+space", "alt+v", "f9".

    Usage:
        listener = GlobalHotkeyListener(
            hotkey="ctrl+shift+space",
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
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

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
        """Stop the listener and unhook the hotkey."""
        try:
            import keyboard

            key = _parse_hotkey(self._hotkey_str)
            if key:
                with contextlib.suppress(Exception):
                    keyboard.remove_hotkey(key)
        except Exception as e:
            logger.debug(f"Error stopping hotkey listener: {e}")
        logger.info("Global hotkey listener stopped")

    def _on_hotkey(self) -> None:
        """Toggle recording state when hotkey is pressed."""
        with self._lock:
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

    def _run(self) -> None:
        """Background thread: register hotkey and wait."""
        key = _parse_hotkey(self._hotkey_str)
        if not key:
            return

        try:
            import keyboard

            # Register the hotkey
            keyboard.add_hotkey(key, self._on_hotkey, suppress=False)
            logger.debug(f"Registered hotkey: {key}")

            # Block until the thread is stopped
            keyboard.wait()

        except ImportError:
            logger.error("keyboard library not installed. Install with: pip install keyboard")
        except Exception as e:
            logger.error(f"Global hotkey listener failed: {e}")
