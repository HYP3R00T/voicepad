"""Floating status overlay for VoicePad global hotkey mode.

A small borderless pill window that appears at the bottom-center of the
screen when the global hotkey is active. Shows recording / transcribing /
copied states. Disappears automatically when idle.

Uses tkinter (stdlib) — no extra dependencies.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Literal

logger = logging.getLogger(__name__)

State = Literal["recording", "transcribing", "copied", "error", "hidden"]

# Pill appearance
_BG = "#1e1e2e"  # Catppuccin Mocha base
_FG = "#cdd6f4"  # foreground
_COLORS: dict[State, str] = {
    "recording": "#f28fad",  # red
    "transcribing": "#fae3b0",  # yellow
    "copied": "#abe9b3",  # green
    "error": "#f28fad",  # red
    "hidden": "#cdd6f4",
}
_LABELS: dict[State, str] = {
    "recording": "● Recording…",
    "transcribing": "◌ Transcribing…",
    "copied": "✓ Copied",
    "error": "✕ Error",
    "hidden": "",
}
_AUTO_HIDE_AFTER_S = 2.0  # hide "Copied" / "Error" after this many seconds


class StatusOverlay:
    """Borderless floating pill shown during hotkey recording sessions.

    Must be created and run on its own thread (Tk mainloop blocks).
    All state changes are thread-safe via after() scheduling.
    """

    def __init__(self) -> None:
        self._root: object | None = None
        self._label: object | None = None
        self._state: State = "hidden"
        self._hide_timer: object | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Tk mainloop in a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="overlay")
        self._thread.start()
        self._ready.wait(timeout=3.0)

    def stop(self) -> None:
        """Destroy the window and stop the mainloop."""
        if self._root is not None:
            with contextlib.suppress(Exception):
                self._root.after(0, self._root.destroy)  # type: ignore[union-attr]

    def set_state(self, state: State) -> None:
        """Update the pill state from any thread."""
        if self._root is None:
            return
        with contextlib.suppress(Exception):
            self._root.after(0, lambda s=state: self._apply_state(s))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            self._root = root

            # Borderless, always-on-top, transparent background
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.92)
            root.configure(bg=_BG)

            # Pill label
            label = tk.Label(
                root,
                text="",
                font=("Segoe UI", 11, "bold"),
                bg=_BG,
                fg=_FG,
                padx=18,
                pady=8,
            )
            label.pack()
            self._label = label

            # Position: bottom-center
            root.update_idletasks()
            self._reposition()

            # Start hidden
            root.withdraw()
            self._ready.set()

            root.mainloop()
        except Exception as e:
            logger.error(f"Overlay failed: {e}")
            self._ready.set()

    def _reposition(self) -> None:
        """Place the pill at the bottom-center of the primary screen."""
        if self._root is None:
            return
        try:
            root = self._root  # type: ignore[assignment]
            root.update_idletasks()
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            w = root.winfo_reqwidth()
            h = root.winfo_reqheight()
            x = (sw - w) // 2
            y = sh - h - 60  # 60px from bottom
            root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def _apply_state(self, state: State) -> None:
        """Apply a new state on the Tk thread."""
        if self._root is None:
            return

        # Cancel any pending auto-hide
        if self._hide_timer is not None:
            with contextlib.suppress(Exception):
                self._root.after_cancel(self._hide_timer)  # type: ignore[union-attr]
            self._hide_timer = None

        self._state = state

        if state == "hidden":
            self._root.withdraw()  # type: ignore[union-attr]
            return

        # Update label text and colour
        text = _LABELS.get(state, "")
        color = _COLORS.get(state, _FG)
        self._label.configure(text=text, fg=color)  # type: ignore[union-attr]
        self._reposition()
        self._root.deiconify()  # type: ignore[union-attr]
        self._root.lift()  # type: ignore[union-attr]

        # Auto-hide transient states
        if state in ("copied", "error"):
            delay_ms = int(_AUTO_HIDE_AFTER_S * 1000)
            self._hide_timer = self._root.after(delay_ms, lambda: self._apply_state("hidden"))  # type: ignore[union-attr]
