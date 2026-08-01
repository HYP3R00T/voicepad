from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Literal

State = Literal["initializing", "ready", "recording", "transcribing", "copied", "error", "hidden"]


@dataclass(frozen=True, slots=True)
class _Presentation:
    summary: str
    body: str
    icon: str
    urgency: str
    timeout_ms: int


_PRESENTATIONS: dict[State, _Presentation] = {
    "initializing": _Presentation(
        "VoicePad is starting",
        "Verifying CUDA and warming Parakeet",
        "view-refresh-symbolic",
        "low",
        0,
    ),
    "ready": _Presentation(
        "VoicePad is ready",
        "Press the global shortcut to start listening",
        "emblem-ok-symbolic",
        "low",
        3000,
    ),
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

    def __init__(self) -> None:
        self._notification_id: int | None = None
        self._state: State = "hidden"
        self._last_elapsed_second: int | None = None
        self._hide_timer: threading.Timer | None = None
        self._generation = 0
        self._lock = threading.Lock()
        self._notify_send = shutil.which("notify-send")
        self._gdbus = shutil.which("gdbus")

    def start(self) -> None:
        """Retained as a no-op lifecycle boundary for the TUI."""

    def stop(self) -> None:
        self.set_state("hidden")

    def set_state(self, state: State) -> None:
        with self._lock:
            self._cancel_hide_timer()
            self._generation += 1
            self._state = state
            self._last_elapsed_second = 0 if state == "recording" else None
            if state == "hidden":
                self._close()
                return
            presentation = _PRESENTATIONS[state]
            if state == "recording":
                presentation = _recording_presentation(0)
            self._show(presentation)
            if presentation.timeout_ms > 0:
                timer = threading.Timer(
                    presentation.timeout_ms / 1000,
                    self._hide_if_current,
                    args=(state, self._generation),
                )
                timer.daemon = True
                self._hide_timer = timer
                timer.start()

    def set_recording_elapsed(self, elapsed_seconds: float) -> None:
        elapsed_second = max(0, int(elapsed_seconds))
        with self._lock:
            if self._state != "recording" or elapsed_second == self._last_elapsed_second:
                return
            self._last_elapsed_second = elapsed_second
            self._show(_recording_presentation(elapsed_second))

    def _hide_if_current(self, expected_state: State, expected_generation: int) -> None:
        with self._lock:
            if self._state != expected_state or self._generation != expected_generation:
                return
            self._hide_timer = None
            self._state = "hidden"
            self._close()

    def _cancel_hide_timer(self) -> None:
        if self._hide_timer is not None:
            self._hide_timer.cancel()
            self._hide_timer = None

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


def _recording_presentation(elapsed_seconds: int) -> _Presentation:
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
    recording = _PRESENTATIONS["recording"]
    return _Presentation(
        summary=f"{recording.summary} · {elapsed}",
        body=recording.body,
        icon=recording.icon,
        urgency=recording.urgency,
        timeout_ms=recording.timeout_ms,
    )
