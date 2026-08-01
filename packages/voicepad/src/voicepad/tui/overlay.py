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

from voicepad.tui.monitors import active_monitors, bottom_center_position, select_monitor

if TYPE_CHECKING:
    import tkinter as tk

logger = logging.getLogger(__name__)

State = Literal["recording", "transcribing", "copied", "error", "hidden"]

# Per-theme colors: (background, foreground, recording, transcribing, copied, error)
_THEME_COLORS: dict[str, tuple[str, str, str, str, str, str]] = {
    "tokyo-night": ("#1A1B26", "#a9b1d6", "#F7768E", "#E0AF68", "#9ECE6A", "#F7768E"),
    "catppuccin-mocha": ("#181825", "#cdd6f4", "#F28FAD", "#FAE3B0", "#ABE9B3", "#F28FAD"),
    "catppuccin-frappe": ("#303446", "#C6D0F5", "#E78284", "#E5C890", "#A6D189", "#E78284"),
    "catppuccin-macchiato": ("#24273A", "#CAD3F5", "#ED8796", "#EED49F", "#A6DA95", "#ED8796"),
    "catppuccin-latte": ("#EFF1F5", "#4C4F69", "#D20F39", "#DF8E1D", "#40A02B", "#D20F39"),
    "dracula": ("#282A36", "#F8F8F2", "#FF5555", "#FFB86C", "#50FA7B", "#FF5555"),
    "nord": ("#2E3440", "#D8DEE9", "#BF616A", "#EBCB8B", "#A3BE8C", "#BF616A"),
    "gruvbox": ("#282828", "#fbf1c7", "#fb4934", "#fe8019", "#b8bb26", "#fb4934"),
    "monokai": ("#272822", "#d6d6d6", "#F92672", "#FD971F", "#A6E22E", "#F92672"),
    "flexoki": ("#100F0F", "#FFFCF0", "#AF3029", "#AD8301", "#66800B", "#AF3029"),
    "solarized-dark": ("#002b36", "#839496", "#dc322f", "#cb4b16", "#859900", "#dc322f"),
    "solarized-light": ("#fdf6e3", "#586e75", "#dc322f", "#cb4b16", "#859900", "#dc322f"),
    "rose-pine": ("#191724", "#e0def4", "#eb6f92", "#f6c177", "#9ccfd8", "#eb6f92"),
    "rose-pine-moon": ("#232136", "#e0def4", "#eb6f92", "#f6c177", "#9ccfd8", "#eb6f92"),
    "rose-pine-dawn": ("#faf4ed", "#575279", "#b4637a", "#ea9d34", "#56949f", "#b4637a"),
    "atom-one-dark": ("#282C34", "#ABB2BF", "#F06262", "#DEB25B", "#62F062", "#F06262"),
    "atom-one-light": ("#FAFAFA", "#383A42", "#F23F3F", "#D8D938", "#6CF23F", "#F23F3F"),
    "textual-dark": ("#121212", "#e0e0e0", "#ba3c5b", "#ffa62b", "#4EBF71", "#ba3c5b"),
    "textual-light": ("#E0E0E0", "#121212", "#ba3c5b", "#ffa62b", "#4EBF71", "#ba3c5b"),
}
_DEFAULT_THEME = "tokyo-night"

_LABELS: dict[State, str] = {
    "recording": "● Recording…",
    "transcribing": "◌ Transcribing…",
    "copied": "✓ Copied",
    "error": "✕ Error",
    "hidden": "",
}
_AUTO_HIDE_AFTER_S = 2.0  # hide "Copied" / "Error" after this many seconds

# Module-level aliases for the default theme — used by tests and external code
_bg, _fg, _c_rec, _c_trans, _c_copy, _c_err = _THEME_COLORS[_DEFAULT_THEME]
_BG = _bg
_FG = _fg
_COLORS: dict[State, str] = {
    "recording": _c_rec,
    "transcribing": _c_trans,
    "copied": _c_copy,
    "error": _c_err,
    "hidden": _fg,
}


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

    def __init__(self, theme: str = _DEFAULT_THEME) -> None:
        colors = _THEME_COLORS.get(theme, _THEME_COLORS[_DEFAULT_THEME])
        self._bg, self._fg, self._c_recording, self._c_transcribing, self._c_copied, self._c_error = colors
        self._state_colors: dict[State, str] = {
            "recording": self._c_recording,
            "transcribing": self._c_transcribing,
            "copied": self._c_copied,
            "error": self._c_error,
            "hidden": self._fg,
        }
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
                self._root.after(0, self._root.quit)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

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
            root.configure(bg=self._bg)

            # Use a Canvas to draw a rounded-rectangle pill shape
            canvas = tk.Canvas(root, bg=self._bg, highlightthickness=0)
            canvas.pack()
            self._canvas = canvas

            # Label drawn over the canvas
            label = tk.Label(
                root,
                text="",
                font=("Segoe UI", 12, "bold"),
                bg=self._bg,
                fg=self._fg,
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
            root.destroy()
        except Exception as e:
            logger.error(f"Overlay failed: {e}")
            self._ready.set()
        finally:
            # Release all Tk object references on this thread so their
            # C-level deallocation (Tcl_DeleteInterp) happens here, not
            # on the main thread during interpreter shutdown — which would
            # cause "Tcl_AsyncDelete: async handler deleted by the wrong
            # thread".
            self._label = None
            self._canvas = None
            self._root = None
            import gc

            gc.collect()

    def _reposition(self) -> None:
        """Place the pill at the bottom-center of the primary screen."""
        if self._root is None:
            return
        try:
            self._root.update_idletasks()
            width = self._root.winfo_reqwidth()
            height = self._root.winfo_reqheight()
            pointer = self._root.winfo_pointerxy()
            monitor = select_monitor(active_monitors(), pointer)
            if monitor is None:
                screen_width = self._root.winfo_screenwidth()
                screen_height = self._root.winfo_screenheight()
                x = (screen_width - width) // 2
                y = screen_height - height - 60
            else:
                x, y = bottom_center_position(monitor, (width, height))
            self._root.geometry(f"{width}x{height}{x:+d}{y:+d}")
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
        color = self._state_colors.get(state, self._fg)

        # Measure text size to size the pill
        self._label.configure(text=text, fg=color, bg=self._bg)
        self._root.update_idletasks()
        lw = self._label.winfo_reqwidth()
        lh = self._label.winfo_reqheight()

        r = lh // 2  # radius = half the height for a true pill
        w = lw
        h = lh

        # Resize canvas and draw rounded rect
        self._canvas.configure(width=w, height=h)
        self._canvas.delete("all")
        _draw_pill(self._canvas, 0, 0, w, h, r, fill=self._bg, outline=color, width=2)

        # Place label centred on canvas
        self._canvas.create_window(w // 2, h // 2, window=self._label)

        self._reposition()
        self._root.deiconify()
        self._root.lift()

        # Auto-hide transient states
        if state in ("copied", "error"):
            delay_ms = int(_AUTO_HIDE_AFTER_S * 1000)
            self._hide_timer = self._root.after(delay_ms, lambda: self._apply_state("hidden"))
