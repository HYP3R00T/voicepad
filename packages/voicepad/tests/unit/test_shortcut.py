from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pytest import MonkeyPatch
from voicepad.tui.shortcut import desktop_shortcut_status


def test_cosmic_shortcut_status_detects_toggle_command(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    command = "/opt/voicepad/bin/voicepad toggle"
    shortcut = tmp_path / ".config/cosmic/com.system76.CosmicSettings.Shortcuts/v1/custom"
    shortcut.parent.mkdir(parents=True)
    shortcut.write_text(f'{{ key: "space" }}: Spawn("{command}")', encoding="utf-8")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "COSMIC")

    with (
        patch("voicepad.tui.shortcut.Path.home", return_value=tmp_path),
        patch("voicepad.tui.shortcut.toggle_command", return_value=command),
    ):
        status = desktop_shortcut_status()

    assert status.desktop == "COSMIC"
    assert status.configured is True
    assert status.command == command
