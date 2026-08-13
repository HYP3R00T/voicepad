import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Input, Label, Markdown, OptionList, ProgressBar, Select, Static, Switch, TabPane
from voicepad.config import AppConfig, load_config
from voicepad.runtime import ApplicationRuntime
from voicepad.tui.app import HistoryEntry, VoicePadApp, _history_label, _recorded_at
from voicepad.tui.components import VoiceButton
from voicepad.tui.setup import SetupModal
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, HuggingFaceSource
from voicepad_core.inference import ActiveDeployment
from voicepad_core.pipeline import TranscriptionProgress


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
        self.ready = True
        self.activation_error: Exception | None = None
        self.activation_count = 0

    def artifacts_ready(self) -> bool:
        return self.ready

    def activate(self, on_progress=None) -> ActiveDeployment:  # type: ignore[no-untyped-def]
        self.activation_count += 1
        if self.activation_error is not None:
            raise self.activation_error
        if on_progress is not None:
            on_progress(100, 200)
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
            app._show_live_update(TranscriptionProgress("provisional words", 1, 26 * 16_000))
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

            app._microphone = microphone = MagicMock()
            microphone.capture_error = RuntimeError("capture failed")
            app._state = "recording"
            with patch.object(app, "_stop_recording") as stop:
                app._update_timer()
            stop.assert_called_once_with()
            assert app._state == "transcribing"

    assert runtime.closed is True
    assert desktop_status.stopped is True
    assert runtime.activation_count == 1


@pytest.mark.asyncio
async def test_missing_artifacts_require_confirmation_before_download(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    runtime.ready = False
    config = AppConfig(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
        artifact_cache_path=tmp_path / "artifacts",
    )
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))

    with (
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            assert "model files are missing" in str(modal.query_one("#setup-reason", Label).render())
            assert runtime.activation_count == 0

            progress = modal.query_one("#setup-progress", ProgressBar)
            progress.display = True
            modal.update_progress(100, 200)
            await pilot.pause()
            assert progress.total == 200
            assert progress.progress == 100
            assert progress.query_one("#bar").region.width == progress.content_region.width

            dialog = modal.query_one("#setup-dialog", Static).content_region
            button = modal.query_one("#setup-continue", VoiceButton).region
            assert abs((button.x + button.width // 2) - (dialog.x + dialog.width // 2)) <= 1

            await pilot.click("#setup-continue")
            await pilot.pause()

            assert not isinstance(app.screen, SetupModal)
            assert app._state == "ready"
            assert runtime.activation_count == 1


@pytest.mark.asyncio
async def test_missing_config_opens_setup_and_saves_defaults(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    config = AppConfig(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
        artifact_cache_path=tmp_path / "artifacts",
    )
    missing_path = tmp_path / "voicepad.toml"

    with (
        patch("voicepad.tui.app.config_path", return_value=missing_path),
        patch("voicepad.tui.app.load_config", return_value=config),
        patch("voicepad.tui.app.save_config") as save,
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
    ):
        app = VoicePadApp(runtime=cast(ApplicationRuntime, runtime))
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            assert "no saved configuration" in str(modal.query_one("#setup-reason", Label).render())
            assert "check for existing" in str(modal.query_one("#setup-explanation", Static).render())
            assert str(modal.query_one("#setup-continue", VoiceButton).label) == "Set up VoicePad"
            assert runtime.activation_count == 0

            await pilot.click("#setup-continue")
            await pilot.pause()

            save.assert_called_once_with(config)
            assert app._config_missing is False
            assert app._state == "ready"


@pytest.mark.asyncio
async def test_setup_failure_stays_visible_and_can_retry(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    runtime.ready = False
    runtime.activation_error = RuntimeError("network unavailable")
    config = AppConfig(artifact_cache_path=tmp_path / "artifacts")
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))

    with (
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#setup-continue")
            await pilot.pause()

            modal = app.screen
            assert isinstance(modal, SetupModal)
            assert "network unavailable" in str(modal.query_one("#setup-status", Label).render())
            assert str(modal.query_one("#setup-continue", VoiceButton).label) == "Retry setup"
            assert modal.query_one("#setup-actions").display is True

            runtime.activation_error = None
            modal.continue_setup()
            await pilot.pause(0.1)

            assert runtime.activation_count == 2
            assert app._state == "ready"


@pytest.mark.asyncio
async def test_settings_write_utilityhub_toml_and_reload_it(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    replacement_runtime = FakeRuntime()
    config_path = tmp_path / "voicepad.toml"
    config = AppConfig(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))

    with (
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
        patch("voicepad.config.config_path", return_value=config_path),
        patch(
            "voicepad.tui.app.ApplicationRuntime",
            return_value=cast(ApplicationRuntime, replacement_runtime),
        ),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            recordings = tmp_path / "new-recordings"
            markdown = tmp_path / "new-markdown"
            artifacts = tmp_path / "new-artifacts"
            app.query_one("#setting-recordings", Input).value = str(recordings)
            app.query_one("#setting-markdown", Input).value = str(markdown)
            app.query_one("#setting-artifacts", Input).value = str(artifacts)
            app.query_one("#setting-prefix", Input).value = "dictation"
            app.query_one("#setting-copy", Switch).value = False
            app.query_one("#setting-theme", Select).value = "nord"

            with patch.object(app, "_activate"):
                app.action_save_settings()

            persisted = load_config(config_path)
            assert persisted == app.config
            assert persisted.recordings_path == recordings
            assert persisted.markdown_path == markdown
            assert persisted.artifact_cache_path == artifacts
            assert persisted.recording_prefix == "dictation"
            assert persisted.copy_complete_text is False
            assert persisted.theme == "nord"
            contents = config_path.read_text(encoding="utf-8")
            assert "schema" not in contents
            assert "v2" not in contents.lower()
            assert runtime.closed is True
            assert app.runtime is replacement_runtime


@pytest.mark.asyncio
async def test_settings_cache_change_uses_setup_gate_before_downloading(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    replacement_runtime = FakeRuntime()
    replacement_runtime.ready = False
    config = AppConfig(
        recordings_path=tmp_path / "recordings",
        markdown_path=tmp_path / "markdown",
        artifact_cache_path=tmp_path / "artifacts",
    )
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))

    with (
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
        patch("voicepad.tui.app.save_config"),
        patch(
            "voicepad.tui.app.ApplicationRuntime",
            return_value=cast(ApplicationRuntime, replacement_runtime),
        ),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            new_cache = tmp_path / "empty-cache"
            app.query_one("#setting-artifacts", Input).value = str(new_cache)

            app.action_save_settings()
            await pilot.pause()

            assert isinstance(app.screen, SetupModal)
            assert app.config.artifact_cache_path == new_cache
            assert replacement_runtime.activation_count == 0


@pytest.mark.asyncio
async def test_settings_reload_reports_runtime_close_failure(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    runtime.close = MagicMock(side_effect=RuntimeError("close failed"))  # type: ignore[method-assign]
    config = AppConfig(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))

    with (
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
        patch("voicepad.tui.app.save_config"),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_save_settings()
            assert "runtime cleanup failed: close failed" in str(app.query_one("#settings-status", Label).render())
            assert app.runtime is runtime


@pytest.mark.asyncio
async def test_settings_reload_reports_reactivation_failure(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    config = AppConfig(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")
    app = VoicePadApp(config, runtime=cast(ApplicationRuntime, runtime))

    with (
        patch("voicepad.tui.app.ControlServer.start"),
        patch("voicepad.tui.app.ControlServer.stop"),
        patch("voicepad.tui.app.DesktopStatus", FakeDesktopStatus),
        patch("voicepad.tui.app.save_config"),
        patch("voicepad.tui.app.ApplicationRuntime") as runtime_type,
    ):
        runtime_type.return_value.artifacts_ready.return_value = True
        runtime_type.return_value.activate.side_effect = RuntimeError("reactivation failed")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_save_settings()
            await pilot.pause()
            assert "activation failed: reactivation failed" in str(app.query_one("#status", Label).render())
            assert app._state == "error"
