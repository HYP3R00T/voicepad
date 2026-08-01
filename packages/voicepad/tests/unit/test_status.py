from __future__ import annotations

import subprocess
from unittest.mock import patch

from voicepad.tui.status import DesktopStatus


def test_desktop_status_replaces_one_native_notification_and_closes_it() -> None:
    completed = subprocess.CompletedProcess([], 0, stdout="42\n", stderr="")
    with (
        patch("voicepad.tui.status.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("voicepad.tui.status.subprocess.run", return_value=completed) as run,
    ):
        status = DesktopStatus()
        status.set_state("recording")
        status.set_state("transcribing")
        status.stop()

    first_command = run.call_args_list[0].args[0]
    replacement_command = run.call_args_list[1].args[0]
    close_command = run.call_args_list[2].args[0]
    assert "--transient" in first_command
    assert "--expire-time=0" in first_command
    assert not any(part.startswith("--replace-id=") for part in first_command)
    assert "--replace-id=42" in replacement_command
    assert "--method=org.freedesktop.Notifications.CloseNotification" in close_command
    assert close_command[-1] == "42"


def test_desktop_status_is_optional_when_notification_client_is_unavailable() -> None:
    with (
        patch("voicepad.tui.status.shutil.which", return_value=None),
        patch("voicepad.tui.status.subprocess.run") as run,
    ):
        status = DesktopStatus()
        status.start()
        status.set_state("recording")
        status.stop()

    assert run.call_args_list == []
