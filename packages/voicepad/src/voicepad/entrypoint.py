from __future__ import annotations

import sys


def main() -> int | None:
    """Dispatch the lightweight toggle command or launch the full CLI."""
    if sys.argv[1:] == ["toggle"]:
        from voicepad.tui.control import run_toggle_command

        return run_toggle_command()

    from voicepad.main import app

    app()
    return None
