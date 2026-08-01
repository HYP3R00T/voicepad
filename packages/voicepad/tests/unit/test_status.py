from __future__ import annotations

import subprocess
from collections.abc import Callable
from unittest.mock import patch

from voicepad.tui.status import DesktopStatus


class FakeTimer:
    instances: list[FakeTimer] = []

    def __init__(self, interval: float, function: Callable[..., None], args: tuple[object, ...]) -> None:
        self.interval = interval
        self.function = function
        self.args = args
        self.daemon = False
        self.started = False
        self.cancelled = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.function(*self.args)


def test_desktop_status_replaces_one_native_notification_and_closes_it() -> None:
    completed = subprocess.CompletedProcess([], 0, stdout="42\n", stderr="")
    with (
        patch("voicepad.tui.status.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("voicepad.tui.status.subprocess.run", return_value=completed) as run,
    ):
        status = DesktopStatus()
        status.set_state("recording")
        status.set_recording_elapsed(0.9)
        status.set_recording_elapsed(12.8)
        status.set_recording_elapsed(12.9)
        status.set_recording_elapsed(3661)
        status.set_state("transcribing")
        status.set_recording_elapsed(3662)
        status.stop()

    first_command = run.call_args_list[0].args[0]
    elapsed_command = run.call_args_list[1].args[0]
    hour_command = run.call_args_list[2].args[0]
    transcribing_command = run.call_args_list[3].args[0]
    close_command = run.call_args_list[4].args[0]
    assert "--transient" in first_command
    assert "--expire-time=0" in first_command
    assert first_command[-2] == "VoicePad is listening · 00:00"
    assert not any(part.startswith("--replace-id=") for part in first_command)
    assert "--replace-id=42" in elapsed_command
    assert elapsed_command[-2] == "VoicePad is listening · 00:12"
    assert hour_command[-2] == "VoicePad is listening · 01:01:01"
    assert transcribing_command[-2] == "Transcribing"
    assert "--method=org.freedesktop.Notifications.CloseNotification" in close_command
    assert close_command[-1] == "42"


def test_transient_status_is_explicitly_closed_without_trusting_desktop_timeout() -> None:
    FakeTimer.instances.clear()
    completed = subprocess.CompletedProcess([], 0, stdout="42\n", stderr="")
    with (
        patch("voicepad.tui.status.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("voicepad.tui.status.subprocess.run", return_value=completed) as run,
        patch("voicepad.tui.status.threading.Timer", FakeTimer),
    ):
        status = DesktopStatus()
        status.set_state("copied")
        stale_timer = FakeTimer.instances[-1]
        status.set_state("recording")
        stale_timer.fire()
        status.set_state("copied")
        current_timer = FakeTimer.instances[-1]
        current_timer.fire()
        status.set_state("ready")
        ready_timer = FakeTimer.instances[-1]
        ready_timer.fire()

    assert stale_timer.interval == 2
    assert stale_timer.started is True
    assert stale_timer.cancelled is True
    assert current_timer.started is True
    assert ready_timer.interval == 3
    assert ready_timer.started is True
    assert len(run.call_args_list) == 6
    assert run.call_args_list[-2].args[0][-2] == "VoicePad is ready"
    assert "--method=org.freedesktop.Notifications.CloseNotification" in run.call_args_list[-1].args[0]


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
