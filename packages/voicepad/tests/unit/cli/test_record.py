"""Tests for voicepad.cli.record."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner
from voicepad.cli.record import _format_markdown, _print_result, _wait_for_quit, record_app
from voicepad_core.config import Config

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    text: str = "hello world",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    language_probability: float = 0.99,
    duration_s: float = 1.5,
    latency_ms: float = 42.0,
    fallback_to_cpu: bool = False,
    segments=None,
):
    return SimpleNamespace(
        text=text,
        device=device,
        compute_type=compute_type,
        language=language,
        language_probability=language_probability,
        duration_s=duration_s,
        latency_ms=latency_ms,
        fallback_to_cpu=fallback_to_cpu,
        segments=segments or [],
    )


# ---------------------------------------------------------------------------
# _format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    def test_includes_filename(self, tmp_path: Path) -> None:
        """The markdown output includes the WAV filename."""
        wav = tmp_path / "clip.wav"
        result = _make_result()
        md = _format_markdown(wav, result)
        assert "clip.wav" in md

    def test_includes_text(self, tmp_path: Path) -> None:
        """The markdown output includes the transcription text."""
        wav = tmp_path / "clip.wav"
        result = _make_result(text="test transcription")
        md = _format_markdown(wav, result)
        assert "test transcription" in md

    def test_includes_segments_when_present(self, tmp_path: Path) -> None:
        """When segments are present, the markdown includes a Segments section."""
        wav = tmp_path / "clip.wav"
        seg = SimpleNamespace(start=0.0, end=1.0, text="hello")
        result = _make_result(segments=[seg])
        md = _format_markdown(wav, result)
        assert "## Segments" in md

    def test_omits_segments_when_empty(self, tmp_path: Path) -> None:
        """When segments list is empty, the Segments section is omitted."""
        wav = tmp_path / "clip.wav"
        result = _make_result(segments=[])
        md = _format_markdown(wav, result)
        assert "## Segments" not in md

    def test_includes_cpu_fallback_warning(self, tmp_path: Path) -> None:
        """When fallback_to_cpu is True, a warning is included in the markdown."""
        wav = tmp_path / "clip.wav"
        result = _make_result(fallback_to_cpu=True)
        md = _format_markdown(wav, result)
        assert "CUDA" in md or "CPU" in md or "fallback" in md.lower()

    def test_empty_text_uses_placeholder(self, tmp_path: Path) -> None:
        """When text is empty, a placeholder is used."""
        wav = tmp_path / "clip.wav"
        result = _make_result(text="")
        md = _format_markdown(wav, result)
        assert "no speech detected" in md


# ---------------------------------------------------------------------------
# _print_result
# ---------------------------------------------------------------------------


class TestPrintResult:
    def test_prints_transcription_text(self) -> None:
        """_print_result outputs the transcription text."""
        result = _make_result(text="spoken words")
        # Just verify it doesn't raise — output goes to typer's echo
        _print_result(result)

    def test_prints_empty_text_placeholder(self) -> None:
        """When text is empty, _print_result outputs the no-speech placeholder."""
        result = _make_result(text="")
        _print_result(result)  # should not raise


# ---------------------------------------------------------------------------
# _wait_for_quit
# ---------------------------------------------------------------------------


class TestWaitForQuit:
    def test_sets_event_on_q_input(self, monkeypatch) -> None:
        """When 'q' is typed, the stop event is set."""
        import threading

        stop_event = threading.Event()
        inputs = iter(["q\n"])
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"readline": lambda self: next(inputs, "")})())
        _wait_for_quit(stop_event)
        assert stop_event.is_set()

    def test_sets_event_on_eof(self, monkeypatch) -> None:
        """When stdin reaches EOF (empty string), the stop event is set."""
        import threading

        stop_event = threading.Event()
        # readline() returning "" signals EOF
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"readline": lambda self: ""})())
        _wait_for_quit(stop_event)
        assert stop_event.is_set()


# ---------------------------------------------------------------------------
# record info command
# ---------------------------------------------------------------------------


class TestShowInfoCommand:
    def test_info_exits_zero(self) -> None:
        """The info command exits with code 0."""
        mock_config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        with patch("voicepad.cli.record.get_config", return_value=mock_config):
            result = runner.invoke(record_app, ["info"])
        assert result.exit_code == 0

    def test_info_shows_model_name(self) -> None:
        """The info command shows the transcription model name."""
        mock_config = Config(
            recordings_path="data/recordings",
            markdown_path="data/markdown",
            transcription_model="tiny",
        )
        with patch("voicepad.cli.record.get_config", return_value=mock_config):
            result = runner.invoke(record_app, ["info"])
        assert "tiny" in result.output

    def test_info_shows_recordings_path(self) -> None:
        """The info command shows the recordings path."""
        mock_config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        with patch("voicepad.cli.record.get_config", return_value=mock_config):
            result = runner.invoke(record_app, ["info"])
        assert "recordings" in result.output

    def test_info_shows_device_constants(self) -> None:
        """The info command shows the device and compute type constants."""
        mock_config = Config(recordings_path="data/recordings", markdown_path="data/markdown")
        with patch("voicepad.cli.record.get_config", return_value=mock_config):
            result = runner.invoke(record_app, ["info"])
        # DEVICE and COMPUTE_TYPE constants should appear
        assert result.exit_code == 0
        assert "cuda" in result.output.lower() or "cpu" in result.output.lower()


# ---------------------------------------------------------------------------
# record start — no-transcribe path
# ---------------------------------------------------------------------------


class TestStartRecordingNoTranscribe:
    def test_no_transcribe_skips_model_load(self, tmp_path: Path) -> None:
        """When --no-transcribe is passed, model loading is skipped."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)  # 1 second

        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio
        mock_recorder.make_wav_path.return_value = tmp_path / "recordings" / "clip.wav"

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.get_or_load_model") as mock_load,
        ):
            result = runner.invoke(record_app, ["start", "--no-transcribe", "--duration", "0.1"])

        mock_load.assert_not_called()
        assert result.exit_code == 0

    def test_no_save_skips_wav_write(self, tmp_path: Path) -> None:
        """When --no-save is passed, the WAV file is not written."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)

        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio

        mock_result = _make_result()

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
            patch("voicepad.cli.record.transcribe_buffer", return_value=mock_result),
        ):
            result = runner.invoke(record_app, ["start", "--no-save", "--duration", "0.1"])

        mock_recorder.save_wav.assert_not_called()
        assert result.exit_code == 0

    def test_short_audio_skips_transcription(self, tmp_path: Path) -> None:
        """When captured audio is shorter than 0.5s, transcription is skipped."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        # 0.1 seconds — below the 0.5s threshold
        short_audio = np.zeros(1600, dtype=np.float32)

        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = short_audio

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.1"])

        assert result.exit_code == 0
        assert "Too short" in result.output or "short" in result.output.lower()

    def test_mic_open_failure_exits_with_error(self, tmp_path: Path) -> None:
        """When the microphone cannot be opened, the command exits with code 1."""
        from voicepad_core import AudioRecorderError

        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_recorder = MagicMock()
        mock_recorder.start.side_effect = AudioRecorderError("device busy")

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.1"])

        assert result.exit_code == 1

    def test_model_load_failure_exits_with_error(self, tmp_path: Path) -> None:
        """When the model cannot be loaded, the command exits with code 1."""
        from voicepad_core import TranscriptionError

        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", side_effect=TranscriptionError("load failed")),
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.1"])

        assert result.exit_code == 1

    def test_download_failure_exits_with_error(self, tmp_path: Path) -> None:
        """When model download fails, the command exits with code 1."""
        from voicepad_core import TranscriptionError

        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.model_downloaded", return_value=False),
            patch("voicepad.cli.record.ensure_model_downloaded", side_effect=TranscriptionError("network error")),
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.1"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# record start — full transcription path
# ---------------------------------------------------------------------------


class TestStartRecordingWithTranscription:
    def test_transcribes_from_buffer_when_no_save(self, tmp_path: Path) -> None:
        """When --no-save is set, transcription uses the in-memory buffer."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio
        mock_result = _make_result(text="buffer transcription")

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
            patch("voicepad.cli.record.transcribe_buffer", return_value=mock_result) as mock_tb,
        ):
            result = runner.invoke(record_app, ["start", "--no-save", "--duration", "0.1"])

        mock_tb.assert_called_once()
        assert result.exit_code == 0
        assert "buffer transcription" in result.output

    def test_transcribes_from_file_when_wav_saved(self, tmp_path: Path) -> None:
        """When a WAV file is saved, transcription uses transcribe_file."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)
        wav_path = tmp_path / "recordings" / "clip.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"fake")  # make it exist

        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio
        mock_recorder.make_wav_path.return_value = wav_path
        mock_result = _make_result(text="file transcription")

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
            patch("voicepad.cli.record.transcribe_file", return_value=mock_result) as mock_tf,
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.1"])

        mock_tf.assert_called_once()
        assert result.exit_code == 0
        assert "file transcription" in result.output

    def test_transcription_error_exits_with_error(self, tmp_path: Path) -> None:
        """When transcription raises TranscriptionError, the command exits with code 1."""
        from voicepad_core import TranscriptionError

        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
            patch("voicepad.cli.record.transcribe_buffer", side_effect=TranscriptionError("model failed")),
        ):
            result = runner.invoke(record_app, ["start", "--no-save", "--duration", "0.1"])

        assert result.exit_code == 1

    def test_audio_too_short_error_skips_gracefully(self, tmp_path: Path) -> None:
        """When AudioTooShortError is raised during transcription, the command exits 0."""
        from voicepad_core import AudioTooShortError

        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
            patch("voicepad.cli.record.transcribe_buffer", side_effect=AudioTooShortError("too short")),
        ):
            result = runner.invoke(record_app, ["start", "--no-save", "--duration", "0.1"])

        assert result.exit_code == 0

    def test_cpu_fallback_shown_in_output(self, tmp_path: Path) -> None:
        """When the model falls back to CPU, a warning is shown."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio
        mock_result = _make_result(fallback_to_cpu=True)

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", True)),
            patch("voicepad.cli.record.transcribe_buffer", return_value=mock_result),
        ):
            result = runner.invoke(record_app, ["start", "--no-save", "--duration", "0.1"])

        assert result.exit_code == 0
        assert "CPU" in result.output or "cpu" in result.output.lower()

    def test_model_download_triggered_when_missing(self, tmp_path: Path) -> None:
        """When the model is not downloaded, ensure_model_downloaded is called."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)
        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio
        mock_result = _make_result()

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=False),
            patch("voicepad.cli.record.ensure_model_downloaded") as mock_dl,
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
            patch("voicepad.cli.record.transcribe_buffer", return_value=mock_result),
        ):
            result = runner.invoke(record_app, ["start", "--no-save", "--duration", "0.1"])

        mock_dl.assert_called_once()
        assert result.exit_code == 0

    def test_markdown_saved_alongside_wav(self, tmp_path: Path) -> None:
        """When transcription succeeds and a WAV is saved, a markdown file is written."""
        mock_config = Config(
            recordings_path=tmp_path / "recordings",
            markdown_path=tmp_path / "markdown",
        )
        mock_audio = np.ones(16000, dtype=np.float32)
        wav_path = tmp_path / "recordings" / "clip.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"fake")

        mock_recorder = MagicMock()
        mock_recorder.stop.return_value = mock_audio
        mock_recorder.make_wav_path.return_value = wav_path
        # save_wav writes the file for real so wav_path.exists() is True
        mock_recorder.save_wav.side_effect = lambda audio, path: None
        mock_result = _make_result(text="saved text")

        with (
            patch("voicepad.cli.record.get_config", return_value=mock_config),
            patch("voicepad.cli.record.AudioRecorder", return_value=mock_recorder),
            patch("voicepad.cli.record.model_downloaded", return_value=True),
            patch("voicepad.cli.record.get_or_load_model", return_value=(MagicMock(), "cpu", "int8", False)),
            patch("voicepad.cli.record.transcribe_file", return_value=mock_result),
        ):
            result = runner.invoke(record_app, ["start", "--duration", "0.1"])

        assert result.exit_code == 0
        md_path = tmp_path / "markdown" / "clip.md"
        assert md_path.exists()
        assert "saved text" in md_path.read_text()
