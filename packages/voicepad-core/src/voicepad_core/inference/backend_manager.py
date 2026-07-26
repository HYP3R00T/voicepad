from __future__ import annotations

from .contracts import BackendDriver
from .errors import BackendLookupError
from .runtime import ActiveRuntimeManager


class BackendRegistry:
    """Registry of backend drivers explicitly enabled by the application."""

    def __init__(self) -> None:
        self._drivers: dict[str, BackendDriver] = {}

    def register(self, driver: BackendDriver) -> None:
        """Register one driver by its stable backend identifier."""
        backend_id = driver.id
        if not backend_id.strip():
            raise ValueError("backend id must not be empty")
        if backend_id in self._drivers:
            raise ValueError(f"Backend '{backend_id}' is already registered.")
        self._drivers[backend_id] = driver

    def get(self, backend_id: str) -> BackendDriver:
        """Return a registered driver or raise an explicit lookup error."""
        try:
            return self._drivers[backend_id]
        except KeyError as exc:
            raise BackendLookupError(f"Backend '{backend_id}' is not registered.") from exc

    def list(self) -> tuple[str, ...]:
        """Return registered backend identifiers in registration order."""
        return tuple(self._drivers)


class SessionManager(ActiveRuntimeManager):
    """Compatibility name for VoicePad's single-active-runtime manager."""


__all__ = ["BackendRegistry", "SessionManager"]
