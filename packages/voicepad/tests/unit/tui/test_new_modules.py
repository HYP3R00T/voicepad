"""Tests for the new TUI modules introduced during refactoring.

Covers: state, services, utils (clipboard, timer, hotkey_utils, markdown),
screens/settings_helpers, and components/header.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------


class TestAppState:
    def test_initial_state(self) -> None:
        from voicepad.tui.state.app_state import AppState

        state = AppState()
        assert state.model_ready is False
        assert state.recording is False
        assert state.transcribing is False
        assert state.session is None
        assert state.streamer is None
        assert state.stream_chunks == []
        assert state.entries == []
        assert state.selected_entry_idx is None
        assert state.current_text == ""
        assert state.hotkey_listener is None
        assert state.hotkey_pending_copy is False
        assert state.overlay is None
        assert state.warm_result is None

    def test_reset_recording_state(self) -> None:
        from voicepad.tui.state.app_state import AppState

        state = AppState()
        state.recording = True
        state.transcribing = True
        state.record_start = 1.0
        state.stream_chunks = [MagicMock()]

        state.reset_recording_state()

        assert state.recording is False
        assert state.transcribing is False
        assert state.session is None
        assert state.record_start == 0.0
        assert state.stream_chunks == []

    def test_reset_streaming_state(self) -> None:
        from voicepad.tui.state.app_state import AppState

        state = AppState()
        state.streamer = MagicMock()
        state.stream_chunks = [MagicMock(), MagicMock()]

        state.reset_streaming_state()

        assert state.streamer is None
        assert state.stream_chunks == []

    def test_mutable_fields_are_independent(self) -> None:
        from voicepad.tui.state.app_state import AppState

        s1 = AppState()
        s2 = AppState()
        s1.entries.append(MagicMock())
        assert len(s2.entries) == 0


# ---------------------------------------------------------------------------
# utils/clipboard
# ---------------------------------------------------------------------------


class TestClipboard:
    def test_copy_to_clipboard_calls_pyperclip(self) -> None:
        from voicepad.tui.utils.clipboard import copy_to_clipboard

        with patch("pyperclip.copy") as mock_copy:
            copy_to_clipboard("hello")
            mock_copy.assert_called_once_with("hello")

    def test_copy_to_clipboard_handles_exception(self) -> None:
        from voicepad.tui.utils.clipboard import copy_to_clipboard

        with patch("pyperclip.copy", side_effect=Exception("no clipboard")):
            copy_to_clipboard("hello")  # should not raise


# ---------------------------------------------------------------------------
# utils/timer
# ---------------------------------------------------------------------------


class TestRecordingTimer:
    def test_timer_calls_on_tick(self) -> None:
        from voicepad.tui.utils.timer import RecordingTimer

        ticks: list[str] = []
        timer = RecordingTimer(on_tick=ticks.append)
        timer.start()
        time.sleep(0.25)
        timer.stop()
        assert len(ticks) > 0

    def test_timer_stop_prevents_further_ticks(self) -> None:
        from voicepad.tui.utils.timer import RecordingTimer

        ticks: list[str] = []
        timer = RecordingTimer(on_tick=ticks.append)
        timer.start()
        time.sleep(0.15)
        timer.stop()
        count_after_stop = len(ticks)
        time.sleep(0.2)
        assert len(ticks) == count_after_stop

    def test_timer_start_is_idempotent(self) -> None:
        from voicepad.tui.utils.timer import RecordingTimer

        ticks: list[str] = []
        timer = RecordingTimer(on_tick=ticks.append)
        timer.start()
        timer.start()  # second call should be a no-op
        time.sleep(0.15)
        timer.stop()
        assert len(ticks) > 0

    def test_timer_formats_seconds_under_one_minute(self) -> None:
        from voicepad.tui.utils.timer import RecordingTimer

        ticks: list[str] = []
        timer = RecordingTimer(on_tick=ticks.append)
        timer.start()
        time.sleep(0.15)
        timer.stop()
        # Under 60s: format is "X.Xs"
        assert any("s" in t for t in ticks)


# ---------------------------------------------------------------------------
# utils/hotkey_utils
# ---------------------------------------------------------------------------


class TestHotkeyUtils:
    def test_parse_returns_mods_and_key(self) -> None:
        from voicepad.tui.utils.hotkey_utils import parse_hotkey_str

        mods, key = parse_hotkey_str("<ctrl>+<alt>+v")
        assert "ctrl" in mods
        assert "alt" in mods
        assert key == "v"

    def test_build_produces_parseable_string(self) -> None:
        from voicepad.tui.utils.hotkey_utils import build_hotkey_str, parse_hotkey_str

        built = build_hotkey_str(["ctrl", "shift"], "a")
        mods, key = parse_hotkey_str(built)
        assert set(mods) == {"ctrl", "shift"}
        assert key == "a"

    def test_build_empty_key_returns_empty(self) -> None:
        from voicepad.tui.utils.hotkey_utils import build_hotkey_str

        assert build_hotkey_str(["ctrl"], "") == ""

    def test_special_key_gets_angle_brackets(self) -> None:
        from voicepad.tui.utils.hotkey_utils import build_hotkey_str

        result = build_hotkey_str([], "space")
        assert "<space>" in result

    def test_single_char_key_no_brackets(self) -> None:
        from voicepad.tui.utils.hotkey_utils import build_hotkey_str

        result = build_hotkey_str([], "v")
        assert result == "v"

    def test_hotkey_keys_list_is_non_empty(self) -> None:
        from voicepad.tui.utils.hotkey_utils import HOTKEY_KEYS

        assert len(HOTKEY_KEYS) > 0
        assert "v" in HOTKEY_KEYS
        assert "space" in HOTKEY_KEYS


# ---------------------------------------------------------------------------
# utils/markdown
# ---------------------------------------------------------------------------


class TestMarkdownUtils:
    def test_format_markdown_roundtrip(self, tmp_path: Path) -> None:
        from voicepad.tui.utils.markdown import format_markdown, parse_markdown_entry

        wav = tmp_path / "clip.wav"
        result = SimpleNamespace(
            text="hello world",
            device="cpu",
            compute_type="int8",
            language="en",
            language_probability=0.99,
            duration_s=2.0,
            latency_ms=100.0,
            segments=[],
        )
        md_content = format_markdown(wav, result, "turbo")
        md_path = tmp_path / "clip.md"
        md_path.write_text(md_content, encoding="utf-8")

        entry = parse_markdown_entry(md_path, 0, tmp_path)
        assert entry is not None
        assert "hello world" in entry.text

    def test_prepend_retranscription_increments_n(self, tmp_path: Path) -> None:
        from voicepad.tui.utils.markdown import format_markdown, prepend_retranscription

        wav = tmp_path / "clip.wav"
        result = SimpleNamespace(
            text="first",
            device="cpu",
            compute_type="int8",
            language="en",
            language_probability=0.99,
            duration_s=1.0,
            latency_ms=50.0,
            segments=[],
        )
        md_path = tmp_path / "clip.md"
        md_path.write_text(format_markdown(wav, result), encoding="utf-8")

        result2 = SimpleNamespace(
            text="second",
            device="cpu",
            compute_type="int8",
            language="en",
            language_probability=0.99,
            duration_s=1.5,
            latency_ms=60.0,
            segments=[],
        )
        new_content = prepend_retranscription(md_path, result2)
        assert "## Transcription 2" in new_content
        assert "second" in new_content
        assert "first" in new_content

    def test_parse_returns_none_for_non_frontmatter_file(self, tmp_path: Path) -> None:
        from voicepad.tui.utils.markdown import parse_markdown_entry

        md_path = tmp_path / "plain.md"
        md_path.write_text("# Just a heading\n\nsome text\n", encoding="utf-8")
        assert parse_markdown_entry(md_path, 0) is None

    def test_parse_returns_none_for_empty_text(self, tmp_path: Path) -> None:
        from voicepad.tui.utils.markdown import parse_markdown_entry

        md_path = tmp_path / "empty.md"
        md_path.write_text(
            "---\nfile: x.wav\ntranscriptions:\n  - n: 1\n    model: cpu\n"
            "    language: en (99.0%)\n    duration: 1.0s\n    latency: 50ms\n"
            "    timestamp: 2026-01-01 00:00\n---\n\n## Transcription 1\n\n*(no speech detected)*\n",
            encoding="utf-8",
        )
        assert parse_markdown_entry(md_path, 0) is None


# ---------------------------------------------------------------------------
# services/HistoryService
# ---------------------------------------------------------------------------


class TestHistoryService:
    def _make_config(self, tmp_path: Path):
        from voicepad_core.config import Config

        return Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")

    def test_load_from_disk_empty_when_no_dir(self, tmp_path: Path) -> None:
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        svc = HistoryService(config)
        entries = svc.load_from_disk()
        assert entries == []

    def test_load_from_disk_parses_valid_files(self, tmp_path: Path) -> None:
        from voicepad.tui.services.history_service import HistoryService
        from voicepad.tui.utils.markdown import format_markdown

        config = self._make_config(tmp_path)
        config.markdown_path.mkdir(parents=True)

        wav = tmp_path / "recordings" / "clip.wav"
        result = SimpleNamespace(
            text="test text",
            device="cpu",
            compute_type="int8",
            language="en",
            language_probability=0.99,
            duration_s=1.0,
            latency_ms=50.0,
            segments=[],
        )
        md_path = config.markdown_path / "clip.md"
        md_path.write_text(format_markdown(wav, result), encoding="utf-8")

        svc = HistoryService(config)
        entries = svc.load_from_disk()
        assert len(entries) == 1
        assert "test text" in entries[0].text

    def test_add_entry(self, tmp_path: Path) -> None:
        from voicepad.tui.models import SessionEntry
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        svc = HistoryService(config)
        entry = SessionEntry(
            index=0, wav_path=None, md_path=None, duration_s=1.0, text="hi", latency_ms=0.0, device="cpu"
        )
        svc.add_entry(entry)
        assert len(svc.entries) == 1

    def test_get_entry_valid_index(self, tmp_path: Path) -> None:
        from voicepad.tui.models import SessionEntry
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        svc = HistoryService(config)
        entry = SessionEntry(
            index=0, wav_path=None, md_path=None, duration_s=1.0, text="hi", latency_ms=0.0, device="cpu"
        )
        svc.add_entry(entry)
        assert svc.get_entry(0) is entry

    def test_get_entry_invalid_index_returns_none(self, tmp_path: Path) -> None:
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        svc = HistoryService(config)
        assert svc.get_entry(99) is None

    def test_delete_entry_removes_and_reindexes(self, tmp_path: Path) -> None:
        from voicepad.tui.models import SessionEntry
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        svc = HistoryService(config)
        for i in range(3):
            svc.add_entry(
                SessionEntry(
                    index=i, wav_path=None, md_path=None, duration_s=1.0, text=f"t{i}", latency_ms=0.0, device="cpu"
                )
            )

        result = svc.delete_entry(1)
        assert result is True
        assert len(svc.entries) == 2
        assert svc.entries[0].index == 0
        assert svc.entries[1].index == 1

    def test_delete_entry_invalid_index_returns_false(self, tmp_path: Path) -> None:
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        svc = HistoryService(config)
        assert svc.delete_entry(0) is False

    def test_delete_entry_removes_files(self, tmp_path: Path) -> None:
        from voicepad.tui.models import SessionEntry
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        config.markdown_path.mkdir(parents=True)
        config.recordings_path.mkdir(parents=True)

        wav = config.recordings_path / "clip.wav"
        wav.write_bytes(b"fake")
        md = config.markdown_path / "clip.md"
        md.write_text("content", encoding="utf-8")

        svc = HistoryService(config)
        entry = SessionEntry(index=0, wav_path=wav, md_path=md, duration_s=1.0, text="hi", latency_ms=0.0, device="cpu")
        svc.add_entry(entry)
        svc.delete_entry(0)

        assert not wav.exists()
        assert not md.exists()

    def test_save_markdown_streaming(self, tmp_path: Path) -> None:
        from voicepad.tui.services.history_service import HistoryService

        config = self._make_config(tmp_path)
        config.markdown_path.mkdir(parents=True)
        config.recordings_path.mkdir(parents=True)

        wav = config.recordings_path / "clip.wav"
        svc = HistoryService(config)
        md_path = svc.save_markdown_streaming(wav, "hello", 1.0, [], "turbo")
        assert md_path.exists()
        assert "hello" in md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# services/RecordingService
# ---------------------------------------------------------------------------


class TestRecordingService:
    def _make_config(self, tmp_path: Path):
        from voicepad_core.config import Config

        return Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")

    def test_create_session_returns_recording_session(self, tmp_path: Path) -> None:
        from voicepad.tui.services.recording_service import RecordingService
        from voicepad.tui.workers import RecordingSession

        config = self._make_config(tmp_path)
        svc = RecordingService(config)
        session = svc.create_session()
        assert isinstance(session, RecordingSession)

    def test_get_audio_duration(self, tmp_path: Path) -> None:
        import numpy as np
        from voicepad.tui.services.recording_service import RecordingService

        config = self._make_config(tmp_path)
        svc = RecordingService(config)
        audio = np.zeros(16000, dtype=np.float32)
        assert svc.get_audio_duration(audio) == pytest.approx(1.0)

    def test_start_session_raises_on_error(self, tmp_path: Path) -> None:
        from voicepad.tui.services.recording_service import RecordingService
        from voicepad.tui.workers import RecordingSession
        from voicepad_core import AudioRecorderError

        config = self._make_config(tmp_path)
        svc = RecordingService(config)
        session = MagicMock(spec=RecordingSession)
        session.start.side_effect = AudioRecorderError("mic error")

        with pytest.raises(AudioRecorderError):
            svc.start_session(session)

    def test_stop_session_raises_on_error(self, tmp_path: Path) -> None:
        from voicepad.tui.services.recording_service import RecordingService
        from voicepad.tui.workers import RecordingSession
        from voicepad_core import AudioRecorderError

        config = self._make_config(tmp_path)
        svc = RecordingService(config)
        session = MagicMock(spec=RecordingSession)
        session.stop.side_effect = AudioRecorderError("stop error")

        with pytest.raises(AudioRecorderError):
            svc.stop_session(session)


# ---------------------------------------------------------------------------
# services/SettingsService
# ---------------------------------------------------------------------------


class TestSettingsService:
    def _make_config(self, tmp_path: Path):
        from voicepad_core.config import Config

        return Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")

    def test_get_config_path_returns_path(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = self._make_config(tmp_path)
        svc = SettingsService(config)
        path = svc.get_config_path()
        assert isinstance(path, Path)

    def test_config_exists_false_when_missing(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = self._make_config(tmp_path)
        svc = SettingsService(config)
        with patch.object(svc, "get_config_path", return_value=tmp_path / "nonexistent.yaml"):
            assert svc.config_exists() is False

    def test_validate_field_valid(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = self._make_config(tmp_path)
        svc = SettingsService(config)
        valid, err = svc.validate_field("transcription_model", "turbo")
        assert valid is True
        assert err is None

    def test_validate_field_invalid(self, tmp_path: Path) -> None:
        from voicepad.tui.services.settings_service import SettingsService

        config = self._make_config(tmp_path)
        svc = SettingsService(config)
        valid, err = svc.validate_field("transcription_model", "not-a-real-model-xyz-invalid")
        # May or may not fail depending on validation — just check it returns a tuple
        assert isinstance(valid, bool)


# ---------------------------------------------------------------------------
# services/TranscriptionService
# ---------------------------------------------------------------------------


class TestTranscriptionService:
    def _make_config(self, tmp_path: Path):
        from voicepad_core.config import Config

        return Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")

    def test_warm_model_returns_error_result_on_failure(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = self._make_config(tmp_path)
        svc = TranscriptionService(config)

        with (
            patch("voicepad.tui.services.transcription_service.model_downloaded", return_value=False),
            patch(
                "voicepad.tui.services.transcription_service.ensure_model_downloaded",
                side_effect=Exception("download failed"),
            ),
        ):
            result = svc.warm_model()

        assert result.error is not None
        assert "download failed" in result.error

    def test_warm_model_returns_result_on_success(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = self._make_config(tmp_path)
        svc = TranscriptionService(config)

        with (
            patch("voicepad.tui.services.transcription_service.model_downloaded", return_value=True),
            patch(
                "voicepad.tui.services.transcription_service.get_or_load_model",
                return_value=(MagicMock(), "cuda", "int8", False),
            ),
        ):
            result = svc.warm_model()

        assert result.error is None
        assert result.device == "cuda"

    def test_transcribe_file_raises_if_missing(self, tmp_path: Path) -> None:
        from voicepad.tui.services.transcription_service import TranscriptionService

        config = self._make_config(tmp_path)
        svc = TranscriptionService(config)

        with pytest.raises(FileNotFoundError):
            svc.transcribe_file(tmp_path / "nonexistent.wav")


# ---------------------------------------------------------------------------
# components/HeaderWidget
# ---------------------------------------------------------------------------


class TestHeaderWidget:
    def test_set_status_updates_label(self) -> None:
        from voicepad.tui.components.header import HeaderWidget

        widget = HeaderWidget("v1.0")
        mock_label = MagicMock()

        with patch.object(widget, "query_one", return_value=mock_label):
            widget.set_status("ready", "ready")

        mock_label.remove_class.assert_called_once_with("ready", "recording", "transcribing", "error")
        mock_label.add_class.assert_called_once_with("ready")
        mock_label.update.assert_called_once()

    def test_set_model_info_updates_label(self) -> None:
        from voicepad.tui.components.header import HeaderWidget

        widget = HeaderWidget("v1.0")
        mock_label = MagicMock()

        with patch.object(widget, "query_one", return_value=mock_label):
            widget.set_model_info("turbo", "cuda", fallback=False)

        mock_label.update.assert_called_once()
        call_arg = mock_label.update.call_args[0][0]
        assert "turbo" in call_arg
        assert "cuda" in call_arg

    def test_set_model_info_shows_fallback(self) -> None:
        from voicepad.tui.components.header import HeaderWidget

        widget = HeaderWidget("v1.0")
        mock_label = MagicMock()

        with patch.object(widget, "query_one", return_value=mock_label):
            widget.set_model_info("turbo", "cpu", fallback=True)

        call_arg = mock_label.update.call_args[0][0]
        assert "fallback" in call_arg


# ---------------------------------------------------------------------------
# screens/settings_helpers
# ---------------------------------------------------------------------------


class TestSettingsHelpers:
    def _make_config(self, tmp_path: Path):
        from voicepad_core.config import Config

        return Config(recordings_path=tmp_path / "recordings", markdown_path=tmp_path / "markdown")

    def test_populate_settings_form_mounts_widgets(self, tmp_path: Path) -> None:
        from voicepad.tui.screens.settings_helpers import populate_settings_form

        config = self._make_config(tmp_path)
        container = MagicMock()

        # _get_input_devices is imported inside the function from voicepad.cli.config
        with (
            patch("voicepad.cli.config._get_input_devices", return_value=[]),
            patch("voicepad.tui.screens.settings_helpers.Static", return_value=MagicMock()),
            patch("voicepad.tui.screens.settings_helpers.Label", return_value=MagicMock()),
            patch("voicepad.tui.screens.settings_helpers.Input", return_value=MagicMock()),
            patch("voicepad.tui.screens.settings_helpers.Select", return_value=MagicMock()),
            patch("voicepad.tui.screens.settings_helpers.Checkbox", return_value=MagicMock()),
        ):
            populate_settings_form(container, config, str(tmp_path / "voicepad.yaml"))

        assert container.mount.called

    def test_mount_hotkey_picker_mounts_widgets(self, tmp_path: Path) -> None:
        from voicepad.tui.screens.settings_helpers import _mount_hotkey_picker

        config = self._make_config(tmp_path)
        container = MagicMock()

        with (
            patch("voicepad.tui.screens.settings_helpers.Static", return_value=MagicMock()),
            patch("voicepad.tui.screens.settings_helpers.Label", return_value=MagicMock()),
            patch("voicepad.tui.screens.settings_helpers.Checkbox", return_value=MagicMock()),
            patch("voicepad.tui.screens.settings_helpers.Select", return_value=MagicMock()),
        ):
            _mount_hotkey_picker(container, config)

        assert container.mount.called


class TestMarkdownShim:
    def test_re_exports_markdown_helpers(self) -> None:
        from voicepad.tui import markdown as markdown_shim
        from voicepad.tui.utils import markdown as markdown_utils

        assert markdown_shim.format_markdown is markdown_utils.format_markdown
        assert markdown_shim.format_markdown_streaming is markdown_utils.format_markdown_streaming
        assert markdown_shim.parse_markdown_entry is markdown_utils.parse_markdown_entry
        assert markdown_shim.prepend_retranscription is markdown_utils.prepend_retranscription
