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

    with patch("voicepad.tui.app.ControlServer.start"), patch("voicepad.tui.app.ControlServer.stop"):
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "ready" in str(app.query_one("#status").render())
            assert "NVIDIA Test GPU" in str(app.query_one("#header-model").render())
            assert app._state == "ready"
            assert app.query_one("#tab-record")
            assert app.query_one("#tab-history")
            assert app.query_one("#tab-settings")
            assert str(app.query_one("#setting-recordings", Input).value) == str(app.config.recordings_path)
            assert app.query_one("#setting-theme", Select).value == app.config.theme

            app._state = "recording"
            app._record_started = time.monotonic() - 2.0
            app._update_timer()
            assert "2.0s" in str(app.query_one("#status").render())

    assert runtime.closed is True
