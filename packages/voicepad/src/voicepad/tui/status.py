from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Literal

State = Literal["recording", "transcribing", "copied", "error", "hidden"]


@dataclass(frozen=True, slots=True)
class _Presentation:
    summary: str
    body: str
    icon: str
    urgency: str
    timeout_ms: int


_PRESENTATIONS: dict[State, _Presentation] = {
    "recording": _Presentation(
        "VoicePad is listening",
        "Press the global shortcut again to stop and transcribe",
        "audio-input-microphone-symbolic",
        "normal",
        0,
    ),
    "transcribing": _Presentation(
        "Transcribing",
        "Finalizing audio and assembling your transcript",
        "emblem-synchronizing-symbolic",
        "normal",
        0,
    ),
    "copied": _Presentation(
        "Transcript copied",
        "Ready to paste",
        "edit-copy-symbolic",
        "low",
        2000,
    ),
    "error": _Presentation(
        "VoicePad needs attention",
        "Open VoicePad for details",
        "dialog-error-symbolic",
        "critical",
        3000,
    ),
    "hidden": _Presentation("", "", "", "low", 0),
}


class DesktopStatus:
    """Replace one transient status through the desktop notification service."""

    def __init__(self, theme: str | None = None) -> None:
        del theme
        self._notification_id: int | None = None
        self._lock = threading.Lock()
        self._notify_send = shutil.which("notify-send")
        self._gdbus = shutil.which("gdbus")

    def start(self) -> None:
        """Retained as a no-op lifecycle boundary for the TUI."""

    def stop(self) -> None:
        self.set_state("hidden")

    def set_state(self, state: State) -> None:
        with self._lock:
            if state == "hidden":
                self._close()
                return
            self._show(_PRESENTATIONS[state])

    def _show(self, presentation: _Presentation) -> None:
        if self._notify_send is None:
            return
        command = [
            self._notify_send,
            "--app-name=VoicePad",
            "--transient",
            "--print-id",
            f"--urgency={presentation.urgency}",
            f"--expire-time={presentation.timeout_ms}",
            f"--icon={presentation.icon}",
        ]
        if self._notification_id is not None:
            command.append(f"--replace-id={self._notification_id}")
        command.extend((presentation.summary, presentation.body))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=True,
                text=True,
                timeout=1,
            )
            self._notification_id = int(completed.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return

    def _close(self) -> None:
        notification_id = self._notification_id
        self._notification_id = None
        if notification_id is None or self._gdbus is None:
            return
        try:
            subprocess.run(
                [
                    self._gdbus,
                    "call",
                    "--session",
                    "--dest=org.freedesktop.Notifications",
                    "--object-path=/org/freedesktop/Notifications",
                    "--method=org.freedesktop.Notifications.CloseNotification",
                    str(notification_id),
                ],
                capture_output=True,
                check=False,
                timeout=1,
            )
        except (OSError, subprocess.SubprocessError):
            return
