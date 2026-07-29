from __future__ import annotations

import contextlib
import logging
import os
import socket
import stat
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_TOGGLE_COMMAND = b"toggle\n"
_OK_RESPONSE = b"ok\n"
_SOCKET_TIMEOUT_S = 1.0


class ControlError(RuntimeError):
    """Base error for the local VoicePad control channel."""


class ControlServer:
    """Accept toggle requests from desktop-managed keyboard shortcuts."""

    def __init__(self, on_toggle: Callable[[], None], socket_path: Path | None = None) -> None:
        self._on_toggle = on_toggle
        self.socket_path = socket_path or get_control_socket_path()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start listening for local toggle requests."""
        if self._server is not None:
            return

        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._remove_stale_socket()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen()
            server.settimeout(0.2)
        except Exception:
            server.close()
            raise

        self._server = server
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._serve, daemon=True, name="voicepad-control")
        self._thread.start()
        logger.info("VoicePad control socket listening: path=%s", self.socket_path)

    def stop(self) -> None:
        """Stop listening and remove the owned socket."""
        server = self._server
        if server is None:
            return

        self._stop_event.set()
        server.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None

        try:
            if self.socket_path.exists() and stat.S_ISSOCK(self.socket_path.stat().st_mode):
                self.socket_path.unlink()
        except OSError as error:
            logger.warning("Could not remove VoicePad control socket '%s': %s", self.socket_path, error)
        logger.info("VoicePad control socket stopped: path=%s", self.socket_path)

    def _remove_stale_socket(self) -> None:
        if not self.socket_path.exists():
            return
        if not stat.S_ISSOCK(self.socket_path.stat().st_mode):
            raise ControlError(f"Control socket path is occupied by a non-socket: {self.socket_path}")

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(_SOCKET_TIMEOUT_S)
        try:
            probe.connect(str(self.socket_path))
        except OSError:
            self.socket_path.unlink()
        else:
            raise ControlError("Another VoicePad instance is already accepting control requests")
        finally:
            probe.close()

    def _serve(self) -> None:
        server = self._server
        if server is None:
            return

        while not self._stop_event.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            with connection:
                try:
                    command = connection.recv(64)
                    if not command:
                        continue
                    if command != _TOGGLE_COMMAND:
                        connection.sendall(b"error: unsupported command\n")
                        continue
                    self._on_toggle()
                    connection.sendall(_OK_RESPONSE)
                except Exception as error:
                    logger.error("VoicePad control request failed: %s", error)
                    with contextlib.suppress(OSError):
                        connection.sendall(b"error: toggle failed\n")


def get_control_socket_path() -> Path:
    """Return the current user's VoicePad control socket path."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "voicepad.sock"
    user_id = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return Path(tempfile.gettempdir()) / f"voicepad-{user_id}.sock"


def request_toggle(socket_path: Path | None = None) -> None:
    """Ask a running VoicePad TUI to toggle recording."""
    path = socket_path or get_control_socket_path()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(_SOCKET_TIMEOUT_S)
    try:
        client.connect(str(path))
        client.sendall(_TOGGLE_COMMAND)
        response = client.recv(64)
    except OSError as error:
        raise ControlError(f"Could not reach the running VoicePad app at {path}: {error}") from error
    finally:
        client.close()

    if response != _OK_RESPONSE:
        message = response.decode("utf-8", errors="replace").strip()
        raise ControlError(message or "VoicePad rejected the toggle request")


def run_toggle_command() -> int:
    """Run the lightweight toggle CLI and return its process exit code."""
    try:
        request_toggle()
    except ControlError as error:
        print(f"VoicePad toggle failed: {error}", file=sys.stderr)
        return 1
    return 0
