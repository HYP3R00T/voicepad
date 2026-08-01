from typing import cast
from unittest.mock import patch

import pytest
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
async def test_tui_activates_resident_nvidia_runtime() -> None:
    runtime = FakeRuntime()
    app = VoicePadApp(AppConfig(), runtime=cast(ApplicationRuntime, runtime))

    with patch("voicepad.tui.app.ControlServer.start"), patch("voicepad.tui.app.ControlServer.stop"):
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "NVIDIA Test GPU" in str(app.query_one("#status").render())
            assert app.query_one("#record").disabled is False

    assert runtime.closed is True
