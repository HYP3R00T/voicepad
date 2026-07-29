from __future__ import annotations

import socket
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from voicepad.tui.control import ControlError, ControlServer, request_toggle, run_toggle_command


class TestControlServer:
    def test_request_toggle_calls_running_server(self, tmp_path: Path) -> None:
        """A client toggle request invokes the running server callback once."""
        on_toggle = MagicMock()
        server = ControlServer(on_toggle=on_toggle, socket_path=tmp_path / "voicepad.sock")
        server.start()

        try:
            request_toggle(server.socket_path)
        finally:
            server.stop()

        on_toggle.assert_called_once_with()

    def test_start_creates_user_only_socket(self, tmp_path: Path) -> None:
        """The control socket is readable and writable only by its owner."""
        server = ControlServer(on_toggle=MagicMock(), socket_path=tmp_path / "voicepad.sock")
        server.start()

        try:
            mode = stat.S_IMODE(server.socket_path.stat().st_mode)
        finally:
            server.stop()

        assert mode == 0o600

    def test_stop_removes_socket(self, tmp_path: Path) -> None:
        """Stopping the server removes its filesystem socket."""
        server = ControlServer(on_toggle=MagicMock(), socket_path=tmp_path / "voicepad.sock")
        server.start()

        server.stop()

        assert not server.socket_path.exists()

    def test_start_replaces_stale_socket(self, tmp_path: Path) -> None:
        """A socket left by a dead process is replaced safely."""
        path = tmp_path / "voicepad.sock"
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()
        server = ControlServer(on_toggle=MagicMock(), socket_path=path)

        server.start()
        try:
            request_toggle(path)
        finally:
            server.stop()

        assert not path.exists()

    def test_start_rejects_non_socket_path(self, tmp_path: Path) -> None:
        """An unrelated file at the control path is preserved and reported."""
        path = tmp_path / "voicepad.sock"
        path.write_text("keep", encoding="utf-8")
        server = ControlServer(on_toggle=MagicMock(), socket_path=path)

        with pytest.raises(ControlError, match="non-socket"):
            server.start()

        assert path.read_text(encoding="utf-8") == "keep"

    def test_start_rejects_second_live_server(self, tmp_path: Path) -> None:
        """A second TUI cannot take over an active control socket."""
        path = tmp_path / "voicepad.sock"
        first = ControlServer(on_toggle=MagicMock(), socket_path=path)
        second = ControlServer(on_toggle=MagicMock(), socket_path=path)
        first.start()

        try:
            with pytest.raises(ControlError, match="Another VoicePad instance"):
                second.start()
        finally:
            first.stop()

    def test_request_toggle_reports_missing_server(self, tmp_path: Path) -> None:
        """A missing TUI returns a clear client error."""
        path = tmp_path / "missing.sock"

        with pytest.raises(ControlError, match="Could not reach the running VoicePad app"):
            request_toggle(path)

    def test_request_toggle_reports_callback_failure(self, tmp_path: Path) -> None:
        """A server callback exception is returned to the client as a failure."""
        server = ControlServer(
            on_toggle=MagicMock(side_effect=RuntimeError("failed")),
            socket_path=tmp_path / "voicepad.sock",
        )
        server.start()

        try:
            with pytest.raises(ControlError, match="toggle failed"):
                request_toggle(server.socket_path)
        finally:
            server.stop()

    def test_invalid_command_is_rejected(self, tmp_path: Path) -> None:
        """Commands outside the minimal toggle protocol are rejected."""
        server = ControlServer(on_toggle=MagicMock(), socket_path=tmp_path / "voicepad.sock")
        server.start()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(1.0)

        try:
            client.connect(str(server.socket_path))
            client.sendall(b"invalid\n")
            response = client.recv(64)
        finally:
            client.close()
            server.stop()

        assert response == b"error: unsupported command\n"

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        """Stopping an inactive server returns immediately."""
        server = ControlServer(on_toggle=MagicMock(), socket_path=tmp_path / "voicepad.sock")

        server.stop()
        time.sleep(0)

        assert not server.socket_path.exists()


class TestRunToggleCommand:
    def test_success_returns_zero(self) -> None:
        """A delivered toggle request returns a successful process status."""
        with patch("voicepad.tui.control.request_toggle") as request_toggle:
            exit_code = run_toggle_command()

        assert exit_code == 0
        request_toggle.assert_called_once_with()

    def test_failure_returns_one_and_reports_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """An unavailable TUI returns failure and writes a diagnostic to stderr."""
        with patch(
            "voicepad.tui.control.request_toggle",
            side_effect=ControlError("not running"),
        ):
            exit_code = run_toggle_command()

        assert exit_code == 1
        assert "VoicePad toggle failed: not running" in capsys.readouterr().err
