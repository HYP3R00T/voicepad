from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pytest
import soundfile as sf
from voicepad_core.audio import (
    AudioConversionDependencyError,
    AudioConversionError,
    AudioFileNotFoundError,
    FileSource,
    RawAudio,
    UnsupportedAudioFormatError,
)


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.touch()
    return path


def test_file_source_rejects_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(AudioFileNotFoundError, match="Audio file not found"):
        FileSource(tmp_path / "missing.wav")

    unsupported = tmp_path / "audio.txt"
    unsupported.touch()
    with pytest.raises(UnsupportedAudioFormatError, match="Unsupported format '.txt'"):
        FileSource(unsupported)


def test_file_source_reads_a_real_wav(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    samples = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    sf.write(path, samples, 16_000, subtype="FLOAT")

    audio = FileSource(path).read_audio()

    npt.assert_array_equal(audio.samples, samples)
    assert audio.sample_rate == 16_000
    assert audio.channels == 1


@pytest.mark.parametrize(
    ("extension", "samples", "sample_rate", "channels"),
    [
        ("wav", np.array([0.1, 0.2], dtype=np.float32), 16_000, 1),
        ("FLAC", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), 44_100, 2),
        ("ogg", np.array([], dtype=np.float32), 22_050, 1),
    ],
)
@patch("voicepad_core.audio.file.sf.read")
def test_native_file_returns_raw_audio(
    soundfile_read: Mock,
    tmp_path: Path,
    extension: str,
    samples: np.ndarray,
    sample_rate: int,
    channels: int,
) -> None:
    path = tmp_path / f"audio.{extension}"
    path.touch()
    soundfile_read.return_value = (samples, sample_rate)

    audio = FileSource(path).read_audio()

    assert isinstance(audio, RawAudio)
    npt.assert_array_equal(audio.samples, samples)
    assert audio.sample_rate == sample_rate
    assert audio.channels == channels
    soundfile_read.assert_called_once_with(str(path), dtype="float32")


@patch("voicepad_core.audio.file.sf.read")
def test_file_source_converts_samples_to_float32(soundfile_read: Mock, wav_path: Path) -> None:
    soundfile_read.return_value = (np.array([100, 200], dtype=np.int16), 16_000)

    audio = FileSource(wav_path).read_audio()

    assert audio.samples.dtype == np.float32
    npt.assert_array_equal(audio.samples, np.array([100, 200], dtype=np.float32))


@patch("voicepad_core.audio.file.sf.read")
def test_file_source_loads_once_and_returns_cached_value(soundfile_read: Mock, wav_path: Path) -> None:
    soundfile_read.return_value = (np.array([0.1, 0.2], dtype=np.float32), 16_000)
    source = FileSource(wav_path)

    first = source.read_audio()
    second = source.read_audio()

    assert first is second
    soundfile_read.assert_called_once_with(str(wav_path), dtype="float32")


@patch("voicepad_core.audio.file.sf.read")
@patch("voicepad_core.audio.file.subprocess.run")
@pytest.mark.parametrize("extension", ["mp3", "M4A", "mp4"])
def test_compressed_file_is_converted_with_ffmpeg(
    subprocess_run: Mock,
    soundfile_read: Mock,
    tmp_path: Path,
    extension: str,
) -> None:
    source_path = tmp_path / f"audio.{extension}"
    source_path.touch()
    samples = np.array([0.1, 0.2], dtype=np.float32)
    soundfile_read.return_value = (samples, 48_000)

    audio = FileSource(source_path).read_audio()

    command = subprocess_run.call_args.args[0]
    assert command[:4] == ["ffmpeg", "-y", "-i", str(source_path)]
    npt.assert_array_equal(audio.samples, samples)
    assert audio.sample_rate == 48_000


@patch("voicepad_core.audio.file.os.remove")
@patch("voicepad_core.audio.file.os.path.exists", return_value=True)
@patch("voicepad_core.audio.file.sf.read")
@patch("voicepad_core.audio.file.subprocess.run")
def test_conversion_temporary_file_is_removed_after_success(
    _subprocess_run: Mock,
    soundfile_read: Mock,
    _path_exists: Mock,
    remove: Mock,
    tmp_path: Path,
) -> None:
    path = tmp_path / "audio.mp3"
    path.touch()
    soundfile_read.return_value = (np.array([0.5], dtype=np.float32), 16_000)

    audio = FileSource(str(path)).read_audio()

    assert audio.frame_count == 1
    remove.assert_called_once()


@patch("voicepad_core.audio.file.subprocess.run", side_effect=FileNotFoundError("missing"))
def test_missing_ffmpeg_is_reported(_subprocess_run: Mock, tmp_path: Path) -> None:
    path = tmp_path / "audio.mp3"
    path.touch()

    with pytest.raises(AudioConversionDependencyError, match="ffmpeg is not installed"):
        FileSource(path).read_audio()


@patch(
    "voicepad_core.audio.file.subprocess.run",
    side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
)
def test_ffmpeg_conversion_failure_is_reported(_subprocess_run: Mock, tmp_path: Path) -> None:
    path = tmp_path / "audio.m4a"
    path.touch()

    with pytest.raises(AudioConversionError, match="ffmpeg failed to convert"):
        FileSource(path).read_audio()


@patch("voicepad_core.audio.file.os.remove")
@patch("voicepad_core.audio.file.os.path.exists", return_value=True)
@patch(
    "voicepad_core.audio.file.subprocess.run",
    side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
)
def test_conversion_temporary_file_is_removed_after_failure(
    _subprocess_run: Mock,
    _path_exists: Mock,
    remove: Mock,
    tmp_path: Path,
) -> None:
    path = tmp_path / "audio.mp4"
    path.touch()

    with pytest.raises(AudioConversionError):
        FileSource(path).read_audio()

    remove.assert_called_once()
