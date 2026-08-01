from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DesktopShortcutStatus:
    desktop: str
    command: str
    configured: bool
    hint: str


def toggle_command() -> str:
    executable = shutil.which("voicepad")
    if executable is None:
        executable = str(Path(sys.executable).with_name("voicepad"))
    return f"{Path(executable).absolute()} toggle"


def desktop_shortcut_status() -> DesktopShortcutStatus:
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "unknown")
    command = toggle_command()
    if "COSMIC" in desktop.upper():
        path = Path.home() / ".config/cosmic/com.system76.CosmicSettings.Shortcuts/v1/custom"
        try:
            configured = command in path.read_text(encoding="utf-8")
        except OSError:
            configured = False
        hint = "COSMIC Keyboard Shortcuts · recommended: Super+Space"
        return DesktopShortcutStatus("COSMIC", command, configured, hint)
    return DesktopShortcutStatus(desktop, command, False, "Bind this command in your desktop keyboard settings")


def open_shortcut_settings() -> None:
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "")
    if "COSMIC" in desktop.upper() and shutil.which("cosmic-settings"):
        subprocess.Popen(
            ["cosmic-settings", "keyboard"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return
    raise RuntimeError("Open your desktop keyboard settings and bind the displayed VoicePad toggle command.")
