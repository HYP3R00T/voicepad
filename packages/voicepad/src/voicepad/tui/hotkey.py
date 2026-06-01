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
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum time (seconds) between modifier press and key press
HOTKEY_TIMEOUT = 0.5


def _parse_hotkey(hotkey_str: str) -> str | None:
    """Return the hotkey string if valid and non-empty, else None."""
    stripped = hotkey_str.strip()
    return stripped if stripped else None


class GlobalHotkeyListener:
    """Listens for a system-wide hotkey and toggles recording state.

    Requires modifiers to be held down while pressing the key (not sequential).
    Includes timeout to prevent delayed triggers.

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

        # Parse hotkey into modifiers and key
        self._required_modifiers: set[Any] = set()
        self._target_key: Any = None
        self._currently_pressed: set[Any] = set()
        self._last_modifier_time: float = 0.0

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

    def _parse_hotkey_components(self, keyboard: Any) -> None:
        """Parse hotkey string into required modifiers and target key."""
        # Parse the hotkey string
        parts = self._hotkey_str.lower().split("+")

        for part in parts:
            part = part.strip().strip("<>")

            # Map modifiers
            if part == "ctrl":
                self._required_modifiers.add(keyboard.Key.ctrl_l)
                self._required_modifiers.add(keyboard.Key.ctrl_r)
            elif part == "alt":
                self._required_modifiers.add(keyboard.Key.alt_l)
                self._required_modifiers.add(keyboard.Key.alt_r)
            elif part == "shift":
                self._required_modifiers.add(keyboard.Key.shift_l)
                self._required_modifiers.add(keyboard.Key.shift_r)
            elif part == "cmd":
                self._required_modifiers.add(keyboard.Key.cmd_l)
                self._required_modifiers.add(keyboard.Key.cmd_r)
            elif part:
                # This is the target key
                if part == "space":
                    self._target_key = keyboard.Key.space
                elif len(part) == 1:
                    self._target_key = keyboard.KeyCode.from_char(part)
                else:
                    # Handle special keys like f1, f2, etc.
                    try:
                        self._target_key = getattr(keyboard.Key, part)
                    except AttributeError:
                        self._target_key = keyboard.KeyCode.from_char(part)

    def _check_modifiers_pressed(self) -> bool:
        """Check if all required modifiers are currently pressed."""
        if not self._required_modifiers:
            return True

        # Check if at least one key from each modifier group is pressed
        modifier_groups = {
            "ctrl": {k for k in self._required_modifiers if "ctrl" in str(k).lower()},
            "alt": {k for k in self._required_modifiers if "alt" in str(k).lower()},
            "shift": {k for k in self._required_modifiers if "shift" in str(k).lower()},
            "cmd": {k for k in self._required_modifiers if "cmd" in str(k).lower()},
        }

        for group_keys in modifier_groups.values():
            if group_keys and not any(k in self._currently_pressed for k in group_keys):
                return False

        return True

    def _on_press(self, key: Any) -> None:
        """Handle key press events."""
        self._currently_pressed.add(key)

        # Update last modifier time when a modifier is pressed
        if key in self._required_modifiers:
            self._last_modifier_time = time.time()

        # Check if this is our target key and all modifiers are pressed
        if key == self._target_key and self._check_modifiers_pressed():
            # Check timeout: ensure modifiers were pressed recently
            time_since_modifier = time.time() - self._last_modifier_time
            if time_since_modifier <= HOTKEY_TIMEOUT:
                self._trigger_hotkey()

    def _on_release(self, key: Any) -> None:
        """Handle key release events."""
        self._currently_pressed.discard(key)

    def _trigger_hotkey(self) -> None:
        """Toggle recording state."""
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
        """Background thread: block on pynput keyboard listener."""
        key = _parse_hotkey(self._hotkey_str)
        if not key:
            return

        try:
            from pynput import keyboard

            # Parse hotkey components
            self._parse_hotkey_components(keyboard)

            # Initialize last modifier time
            self._last_modifier_time = time.time()

            # Create listener with press and release handlers
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()
            self._listener.join()

        except Exception as e:
            logger.error(f"Global hotkey listener failed: {e}")
