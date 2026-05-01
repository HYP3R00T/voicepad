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
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import tkinter as tk

logger = logging.getLogger(__name__)

State = Literal["recording", "transcribing", "copied", "error", "hidden"]

# Pill appearance — dark background with bright text for visibility on dark themes
_BG = "#1e1e2e"  # Catppuccin Mocha base (dark)
_FG = "#cdd6f4"  # Catppuccin Mocha text (light)
_COLORS: dict[State, str] = {
    "recording": "#f38ba8",  # Catppuccin Mocha red (brighter)
    "transcribing": "#f9e2af",  # Catppuccin Mocha yellow (brighter)
    "copied": "#a6e3a1",  # Catppuccin Mocha green (brighter)
    "error": "#f38ba8",  # Catppuccin Mocha red (brighter)
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


def _draw_pill(canvas_obj: object, x1: int, y1: int, x2: int, y2: int, r: int, **kw: object) -> None:
    """Draw a rounded rectangle (pill) on a tkinter Canvas."""
    import tkinter as tk

    canvas = cast(Any, canvas_obj)
    fill = str(kw.get("fill", ""))
    outline = str(kw.get("outline", ""))
    lw = kw.get("width", 1)
    # Filled body (two overlapping rectangles)
    canvas.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline=fill)
    canvas.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline=fill)
    # Four filled corner circles
    canvas.create_oval(x1, y1, x1 + 2 * r, y1 + 2 * r, fill=fill, outline=fill)
    canvas.create_oval(x2 - 2 * r, y1, x2, y1 + 2 * r, fill=fill, outline=fill)
    canvas.create_oval(x1, y2 - 2 * r, x1 + 2 * r, y2, fill=fill, outline=fill)
    canvas.create_oval(x2 - 2 * r, y2 - 2 * r, x2, y2, fill=fill, outline=fill)
    # Border arcs
    canvas.create_arc(x1, y1, x1 + 2 * r, y1 + 2 * r, start=90, extent=90, style=tk.ARC, outline=outline, width=lw)
    canvas.create_arc(x2 - 2 * r, y1, x2, y1 + 2 * r, start=0, extent=90, style=tk.ARC, outline=outline, width=lw)
    canvas.create_arc(x1, y2 - 2 * r, x1 + 2 * r, y2, start=180, extent=90, style=tk.ARC, outline=outline, width=lw)
    canvas.create_arc(x2 - 2 * r, y2 - 2 * r, x2, y2, start=270, extent=90, style=tk.ARC, outline=outline, width=lw)
    # Straight border edges
    canvas.create_line(x1 + r, y1, x2 - r, y1, fill=outline, width=lw)
    canvas.create_line(x1 + r, y2, x2 - r, y2, fill=outline, width=lw)
    canvas.create_line(x1, y1 + r, x1, y2 - r, fill=outline, width=lw)
    canvas.create_line(x2, y1 + r, x2, y2 - r, fill=outline, width=lw)


class StatusOverlay:
    """Borderless floating pill shown during hotkey recording sessions.

    Must be created and run on its own thread (Tk mainloop blocks).
    All state changes are thread-safe via after() scheduling.
    """

    def __init__(self) -> None:
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._label: tk.Label | None = None
        self._state: State = "hidden"
        self._hide_timer: str | None = None
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
                self._root.after(0, self._root.destroy)

    def set_state(self, state: State) -> None:
        """Update the pill state from any thread."""
        if self._root is None:
            return
        with contextlib.suppress(Exception):

            def _update() -> None:
                self._apply_state(state)

            self._root.after(0, _update)

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

            # Use a Canvas to draw a rounded-rectangle pill shape
            canvas = tk.Canvas(root, bg=_BG, highlightthickness=0)
            canvas.pack()
            self._canvas = canvas

            # Label drawn over the canvas
            label = tk.Label(
                root,
                text="",
                font=("Segoe UI", 12, "bold"),
                bg=_BG,
                fg=_FG,
                padx=20,
                pady=9,
            )
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
            self._root.update_idletasks()
            sw = self._root.winfo_screenwidth()
            sh = self._root.winfo_screenheight()
            w = self._root.winfo_reqwidth()
            h = self._root.winfo_reqheight()
            x = (sw - w) // 2
            y = sh - h - 60  # 60px from bottom
            self._root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def _apply_state(self, state: State) -> None:
        """Apply a new state on the Tk thread."""
        if self._root is None or self._label is None or self._canvas is None:
            return

        # Cancel any pending auto-hide
        if self._hide_timer is not None:
            with contextlib.suppress(Exception):
                self._root.after_cancel(self._hide_timer)
            self._hide_timer = None

        self._state = state

        if state == "hidden":
            self._root.withdraw()
            return

        text = _LABELS.get(state, "")
        color = _COLORS.get(state, _FG)

        # Measure text size to size the pill
        self._label.configure(text=text, fg=color, bg=_BG)
        self._root.update_idletasks()
        lw = self._label.winfo_reqwidth()
        lh = self._label.winfo_reqheight()

        r = lh // 2  # radius = half the height for a true pill
        w = lw
        h = lh

        # Resize canvas and draw rounded rect
        self._canvas.configure(width=w, height=h)
        self._canvas.delete("all")
        _draw_pill(self._canvas, 0, 0, w, h, r, fill=_BG, outline=color, width=2)

        # Place label centred on canvas
        self._canvas.create_window(w // 2, h // 2, window=self._label)

        self._reposition()
        self._root.deiconify()
        self._root.lift()

        # Auto-hide transient states
        if state in ("copied", "error"):
            delay_ms = int(_AUTO_HIDE_AFTER_S * 1000)
            self._hide_timer = self._root.after(delay_ms, lambda: self._apply_state("hidden"))
