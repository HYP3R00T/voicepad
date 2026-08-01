from __future__ import annotations

import stat
from pathlib import Path
from threading import Event

from voicepad.tui.control import ControlServer, request_toggle


def test_user_only_control_socket_dispatches_desktop_toggle(tmp_path: Path) -> None:
    toggled = Event()
    socket_path = tmp_path / "voicepad.sock"
    server = ControlServer(toggled.set, socket_path)

    server.start()
    try:
        request_toggle(socket_path)
        assert toggled.wait(timeout=1)
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
    finally:
        server.stop()

    assert not socket_path.exists()
