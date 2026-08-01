import time
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from textual.widgets import Input, Select
from voicepad.config import AppConfig
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.app import VoicePadApp
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import GrowingTranscriptionUpdate


class FakeDesktopStatus:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def set_state(self, state: str) -> None:
        self.states.append(state)


class FakeRuntime:
    def __init__(self) -> None:
        source = PARAKEET_V3_MANIFEST.source
        assert isinstance(source, HuggingFaceSource)
        self.active = ActiveDeployment(
            PARAKEET_V3_CUDA,
            source.revision,
            "GPU-test",
            "NVIDIA Test GPU",
            4_000_000_000,
        )
        self.closed = False

    def activate(self) -> ActiveDeployment:
        return self.active

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_tui_activates_resident_nvidia_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    config = AppConfig(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))

    with (
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "ready" in str(app.query_one("#status").render())
            assert "NVIDIA CUDA" in str(app.query_one("#header-model").render())
            assert app._state == "ready"
            assert app.query_one("#tab-record")
            assert app.query_one("#tab-history")
            assert app.query_one("#tab-settings")
            assert str(app.query_one("#setting-recordings", Input).value) == str(app.config.recordings_path)
            assert app.query_one("#setting-theme", Select).value == app.config.theme

            app._state = "recording"
            app._show_live_update(GrowingTranscriptionUpdate("provisional words", 1, 26 * 16_000))
            assert "provisional words" in str(app.query_one("#tx-text").render())
            assert "through 26.0s" in str(app.query_one("#tx-meta").render())

            app._record_started = time.monotonic() - 2.0
            app._update_timer()
            assert "2.0s" in str(app.query_one("#status").render())

            desktop_status = cast(FakeDesktopStatus, app._desktop_status)
            assert desktop_status.states == ["initializing", "ready"]

            app._state = "ready"
            with patch.object(app, "_start_recording") as start:
                app._external_toggle()
            assert start.called
            assert desktop_status.started is True
            assert desktop_status.states == ["initializing", "ready", "recording"]
            assert app._state == "starting"

    assert runtime.closed is True
    assert desktop_status.stopped is True
