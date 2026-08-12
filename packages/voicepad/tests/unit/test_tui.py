import logging
import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pytest import LogCaptureFixture
from textual.widgets import Input, Label, Markdown, OptionList, Select, Static, Switch, TabPane
from voicepad.config import AppConfig
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.app import HistoryEntry, VoicePadApp, _history_label, _recorded_at
from voicepad_core.audio import CaptureFailure, WavArtifact
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import FileTranscriptionResult, GrowingTranscriptionUpdate


def test_history_entries_have_compact_human_readable_labels(tmp_path: Path) -> None:
    markdown = tmp_path / "daily_note_20260802_120053_314883_8c5b4a12.md"
    entry = HistoryEntry(markdown, None, 30.5, True, "hello")

    assert _recorded_at(markdown.stem) is not None
    assert _history_label(entry) == "✓  Aug 02  12:00:53    30.5s"


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
    config.markdown_path.mkdir()
    (config.markdown_path / "recording_20260802_120053_314883_8c5b4a12.md").write_text(
        "---\naudio: recording.wav\nduration_seconds: 30.5\ncomplete: true\n---\n\nhello from history\n"
    )
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
            assert app.query_one("#setting-copy", Switch).value == app.config.copy_complete_text
            assert len(app.query(".settings-group")) == 3
            assert str(app.query_one("#shortcut-command", Static).render()) == app._shortcut.command
            history_options = app.query_one("#history-options", OptionList)
            assert history_options.option_count == 1
            assert history_options.highlighted == 0
            assert "Aug 02  12:00:53" in str(history_options.get_option_at_index(0).prompt)
            assert "August 02" in str(app.query_one("#history-detail-title", Label).render())
            assert app.query_one("#history-viewer", Markdown).source == "hello from history"
            header = app.query_one("#header", Static)
            assert header.region.height == 2
            assert header.styles.padding.top == 1
            assert app.native_ansi_color is True
            assert app.styles.background.ansi == -1
            assert app.screen.styles.background.a == 0
            assert app.query_one("#tab-record", TabPane).styles.background.a == 0
            assert app.query_one("#tab-history", TabPane).styles.background.a == 0
            assert app.query_one("#tab-settings", TabPane).styles.background.a == 0
            assert app.query_one("#history-list-pane", Static).styles.background.a == 0
            assert app.query_one("#history-detail", Static).styles.background.a == 0
            assert app.query_one("#history-options", OptionList).styles.background.a == 0

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

            failed_microphone = MagicMock()
            failed_microphone.capture_failures = (CaptureFailure("capture-write", RuntimeError("writer failed")),)
            app._microphone = failed_microphone
            app._state = "recording"
            app._record_started = time.monotonic() - 3.0
            with patch.object(app, "_stop_recording") as stop:
                app._update_timer()
            stop.assert_called_once_with()
            assert app._state == "transcribing"
            assert "capture failed" in str(app.query_one("#status").render())

    assert runtime.closed is True
    assert desktop_status.stopped is True


def test_tui_shutdown_uses_authoritative_recording_finalization(tmp_path: Path) -> None:
    runtime = MagicMock()
    config = AppConfig(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))
    microphone = MagicMock()
    job = MagicMock()
    artifact = WavArtifact(tmp_path / "recording.wav", 16_000, 1, 16_000, 1.0)
    source = PARAKEET_V3_MANIFEST.source
    assert isinstance(source, HuggingFaceSource)
    active = ActiveDeployment(PARAKEET_V3_CUDA, source.revision, "GPU-test", "NVIDIA Test GPU", 4_000_000_000)
    result = FileTranscriptionResult("partial", (), (), 1.0, 0.1, active, (), (), (), (), False)
    runtime.stop_recording.return_value = (artifact, result)
    app._microphone = microphone
    app._job = job
    markdown = config.markdown_path / "recording.md"

    with (
        patch.object(app._control, "stop"),
        patch("voicepad.tui.app.persist_markdown", return_value=markdown) as persist,
    ):
        app.on_unmount()

    runtime.stop_recording.assert_called_once_with(microphone, job)
    persist.assert_called_once_with(artifact.path, result, config.markdown_path)
    job.cancel.assert_not_called()
    runtime.close.assert_called_once_with()


def test_tui_shutdown_reports_finalization_and_release_failures(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    runtime = MagicMock()
    app = VoicePadApp(AppConfig(), runtime=cast(ApplicationRuntime, runtime))
    microphone = MagicMock()
    job = MagicMock()
    runtime.stop_recording.side_effect = RuntimeError("finalization failed")
    job.finish.side_effect = RuntimeError("worker did not stop")
    app._microphone = microphone
    app._job = job

    with patch.object(app._control, "stop"):
        app.on_unmount()

    job.cancel.assert_called_once_with()
    assert "TUI shutdown could not finalize the active recording" in caplog.messages
    assert "TUI shutdown could not release the recording job" in caplog.messages
    assert "TUI shutdown finished: failures=2" in caplog.messages
