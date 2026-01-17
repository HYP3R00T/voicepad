from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static, Tab, Tabs


def get_app_version() -> str:
    """Resolve the package version from installation metadata or pyproject.toml."""
    try:
        return version("voicepad")
    except PackageNotFoundError:
        # Fall back to reading pyproject.toml if available
        try:
            pyproject = Path(__file__).parents[2] / "pyproject.toml"
            if pyproject.exists():
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                return data.get("project", {}).get("version", "unknown")
        except Exception:
            pass
    return "unknown"


def get_app_name() -> str:
    """Resolve the application name from pyproject or default to 'Voicepad'."""
    try:
        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        if pyproject.exists():
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            name = data.get("project", {}).get("name")
            if name:
                return str(name)
    except Exception:
        pass
    return "Voicepad"


class AppHeader(Widget):
    """Application header with title and navigation tabs."""

    DEFAULT_CSS = """
    AppHeader {
        height: 3;
        padding: 1;
    }
    AppHeader #app-title {
        padding-right: 2;
        width: auto;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the header with app title and tabs."""
        with Horizontal():
            app_title = f"{get_app_name()} v{get_app_version()}"
            yield Static(app_title, id="app-title")
            tabs = Tabs(
                Tab("Main", id="main"),
                Tab("Config", id="config"),
                id="tabs",
            )
            yield tabs
