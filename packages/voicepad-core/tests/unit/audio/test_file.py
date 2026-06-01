"""
Comprehensive tests for FileSource class.

Tests cover:
- Validation (4 tests)
- Native format loading (6 tests)
- FFmpeg integration (5 tests)
- Lazy loading (4 tests)

Target: 70%+ coverage for file.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pytest
import soundfile as sf
from voicepad_core.audio.file import SUPPORTED_FORMATS, FileSource

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_wav_file(tmp_path: Path) -> Path:
    """Create a temporary WAV file for testing."""
    file_path = tmp_path / "test.wav"
    # Create simple mono audio
    audio_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    sf.write(str(file_path), audio_data, 16000, format="WAV")
    return file_path


@pytest.fixture
def sample_flac_file(tmp_path: Path) -> Path:
    """Create a temporary FLAC file for testing."""
    file_path = tmp_path / "test.flac"
    audio_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    sf.write(str(file_path), audio_data, 16000, format="FLAC")
    return file_path


@pytest.fixture
def sample_ogg_file(tmp_path: Path) -> Path:
    """Create a temporary OGG file for testing."""
    file_path = tmp_path / "test.ogg"
    audio_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    sf.write(str(file_path), audio_data, 16000, format="OGG")
    return file_path


# ============================================================================
# Validation Tests (4 tests)
# ============================================================================


def test_file_not_found() -> None:
    """Test file not found raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        FileSource("nonexistent.wav")


def test_unsupported_format(tmp_path: Path) -> None:
    """Test unsupported format raises ValueError."""
    # Create a text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("not audio")

    with pytest.raises(ValueError, match="Unsupported format '.txt'"):
        FileSource(text_file)


def test_supported_formats_list() -> None:
    """Test SUPPORTED_FORMATS constant contains expected formats."""
    expected_formats = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".mp4"}
    assert expected_formats == SUPPORTED_FORMATS


def test_case_insensitive_extension_matching(tmp_path: Path) -> None:
    """Test case-insensitive extension matching."""
    # Create files with uppercase extensions
    for ext in ["WAV", "FLAC", "OGG"]:
        file_path = tmp_path / f"test.{ext}"
        audio_data = np.array([0.1, 0.2], dtype=np.float32)
        sf.write(str(file_path), audio_data, 16000)

        # Should not raise ValueError
        source = FileSource(file_path)
        assert source._file_path.suffix.lower() in SUPPORTED_FORMATS


# ============================================================================
# Native Format Loading Tests (6 tests)
# ============================================================================


@patch("voicepad_core.audio.file.sf.read")
def test_load_wav_mono(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test WAV file loading (mono)."""
    # Mock soundfile.read to return mono audio
    mock_audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    source = FileSource(sample_wav_file)
    result = source.read()

    # Verify soundfile.read was called
    mock_sf_read.assert_called_once_with(str(sample_wav_file), dtype="float32")

    # Verify result
    assert result.dtype == np.float32
    assert result.ndim == 1
    npt.assert_array_equal(result, mock_audio)
    assert source.get_sample_rate() == 16000
    assert source.get_channels() == 1


@patch("voicepad_core.audio.file.sf.read")
def test_load_wav_stereo(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test WAV file loading (stereo)."""
    # Mock soundfile.read to return stereo audio (N, 2)
    mock_audio = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 44100)

    source = FileSource(sample_wav_file)
    result = source.read()

    # Verify result
    assert result.dtype == np.float32
    assert result.ndim == 2
    assert result.shape == (3, 2)
    npt.assert_array_equal(result, mock_audio)
    assert source.get_sample_rate() == 44100
    assert source.get_channels() == 2


@patch("voicepad_core.audio.file.sf.read")
def test_load_flac(mock_sf_read: Mock, sample_flac_file: Path) -> None:
    """Test FLAC file loading."""
    mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 48000)

    source = FileSource(sample_flac_file)
    result = source.read()

    # Verify soundfile.read was called
    mock_sf_read.assert_called_once_with(str(sample_flac_file), dtype="float32")

    # Verify result
    npt.assert_array_equal(result, mock_audio)
    assert source.get_sample_rate() == 48000
    assert source.get_channels() == 1


@patch("voicepad_core.audio.file.sf.read")
def test_load_ogg(mock_sf_read: Mock, sample_ogg_file: Path) -> None:
    """Test OGG file loading."""
    mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 22050)

    source = FileSource(sample_ogg_file)
    result = source.read()

    # Verify soundfile.read was called
    mock_sf_read.assert_called_once_with(str(sample_ogg_file), dtype="float32")

    # Verify result
    npt.assert_array_equal(result, mock_audio)
    assert source.get_sample_rate() == 22050
    assert source.get_channels() == 1


@patch("voicepad_core.audio.file.sf.read")
def test_channel_detection_1d(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test channel detection for 1D array (mono)."""
    mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    source = FileSource(sample_wav_file)
    source.read()

    assert source.get_channels() == 1


@patch("voicepad_core.audio.file.sf.read")
def test_channel_detection_2d(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test channel detection for 2D array (multi-channel)."""
    # Test various channel counts
    for num_channels in [2, 4, 6]:
        mock_audio = np.random.randn(100, num_channels).astype(np.float32)
        mock_sf_read.return_value = (mock_audio, 44100)

        source = FileSource(sample_wav_file)
        source.read()

        assert source.get_channels() == num_channels


# ============================================================================
# FFmpeg Integration Tests (5 tests)
# ============================================================================


@patch("voicepad_core.audio.file.sf.read")
@patch("voicepad_core.audio.file.subprocess.run")
def test_mp3_conversion_via_ffmpeg(mock_subprocess: Mock, mock_sf_read: Mock, tmp_path: Path) -> None:
    """Test MP3 conversion via ffmpeg."""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.touch()  # Create empty file

    # Mock subprocess.run (ffmpeg)
    mock_subprocess.return_value = Mock(returncode=0)

    # Mock soundfile.read (reading converted WAV)
    mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 44100)

    source = FileSource(mp3_file)
    result = source.read()

    # Verify ffmpeg was called
    mock_subprocess.assert_called_once()
    call_args = mock_subprocess.call_args[0][0]
    assert call_args[0] == "ffmpeg"
    assert "-y" in call_args
    assert "-i" in call_args
    assert str(mp3_file) in call_args

    # Verify result
    npt.assert_array_equal(result, mock_audio)


@patch("voicepad_core.audio.file.sf.read")
@patch("voicepad_core.audio.file.subprocess.run")
def test_m4a_conversion_via_ffmpeg(mock_subprocess: Mock, mock_sf_read: Mock, tmp_path: Path) -> None:
    """Test M4A conversion via ffmpeg."""
    m4a_file = tmp_path / "test.m4a"
    m4a_file.touch()

    mock_subprocess.return_value = Mock(returncode=0)
    mock_audio = np.array([0.1, 0.2], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 48000)

    source = FileSource(m4a_file)
    result = source.read()

    # Verify ffmpeg was called
    mock_subprocess.assert_called_once()
    call_args = mock_subprocess.call_args[0][0]
    assert call_args[0] == "ffmpeg"
    assert str(m4a_file) in call_args

    # Verify result
    npt.assert_array_equal(result, mock_audio)


@patch("voicepad_core.audio.file.sf.read")
@patch("voicepad_core.audio.file.subprocess.run")
def test_mp4_conversion_via_ffmpeg(mock_subprocess: Mock, mock_sf_read: Mock, tmp_path: Path) -> None:
    """Test MP4 conversion via ffmpeg."""
    mp4_file = tmp_path / "test.mp4"
    mp4_file.touch()

    mock_subprocess.return_value = Mock(returncode=0)
    mock_audio = np.array([0.1, 0.2], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 44100)

    source = FileSource(mp4_file)
    result = source.read()

    # Verify ffmpeg was called
    mock_subprocess.assert_called_once()
    npt.assert_array_equal(result, mock_audio)


@patch("voicepad_core.audio.file.subprocess.run")
def test_ffmpeg_failure_raises_runtime_error(mock_subprocess: Mock, tmp_path: Path) -> None:
    """Test ffmpeg failure raises RuntimeError."""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.touch()

    # Mock ffmpeg failure
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

    source = FileSource(mp3_file)

    with pytest.raises(RuntimeError, match="ffmpeg failed to convert"):
        source.read()


@patch("voicepad_core.audio.file.os.remove")
@patch("voicepad_core.audio.file.os.path.exists")
@patch("voicepad_core.audio.file.sf.read")
@patch("voicepad_core.audio.file.subprocess.run")
def test_temp_file_cleanup_on_error(
    mock_subprocess: Mock, mock_sf_read: Mock, mock_exists: Mock, mock_remove: Mock, tmp_path: Path
) -> None:
    """Test temp file cleanup even on error."""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.touch()

    # Mock ffmpeg failure
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
    mock_exists.return_value = True

    source = FileSource(mp3_file)

    with pytest.raises(RuntimeError):
        source.read()

    # Verify cleanup was attempted
    mock_exists.assert_called()
    mock_remove.assert_called_once()


# ============================================================================
# Lazy Loading Tests (4 tests)
# ============================================================================


@patch("voicepad_core.audio.file.sf.read")
def test_lazy_loading(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test audio not loaded until read() called."""
    mock_audio = np.array([0.1, 0.2], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    source = FileSource(sample_wav_file)

    # Verify _audio is None before read()
    assert source._audio is None
    assert mock_sf_read.call_count == 0

    # Call read()
    result = source.read()

    # Verify _audio is populated after read()
    assert source._audio is not None
    npt.assert_array_equal(result, mock_audio)
    assert mock_sf_read.call_count == 1


@patch("voicepad_core.audio.file.sf.read")
def test_get_sample_rate_triggers_load(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test get_sample_rate() triggers load."""
    mock_audio = np.array([0.1, 0.2], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 48000)

    source = FileSource(sample_wav_file)

    # Verify not loaded yet
    assert source._audio is None

    # Call get_sample_rate()
    sample_rate = source.get_sample_rate()

    # Verify file was loaded
    assert source._audio is not None
    assert sample_rate == 48000
    assert mock_sf_read.call_count == 1


@patch("voicepad_core.audio.file.sf.read")
def test_get_channels_triggers_load(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test get_channels() triggers load."""
    mock_audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 44100)

    source = FileSource(sample_wav_file)

    # Verify not loaded yet
    assert source._audio is None

    # Call get_channels()
    channels = source.get_channels()

    # Verify file was loaded
    assert source._audio is not None
    assert channels == 2
    assert mock_sf_read.call_count == 1


@patch("voicepad_core.audio.file.sf.read")
def test_multiple_read_calls_cached(mock_sf_read: Mock, sample_wav_file: Path, capsys) -> None:
    """Test multiple read() calls return same data (cached)."""
    mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    source = FileSource(sample_wav_file)

    # Call read() multiple times
    result1 = source.read()
    result2 = source.read()
    result3 = source.read()

    # Verify file was only loaded once
    assert mock_sf_read.call_count == 1

    # Verify all results are the same
    npt.assert_array_equal(result1, mock_audio)
    npt.assert_array_equal(result2, mock_audio)
    npt.assert_array_equal(result3, mock_audio)


# ============================================================================
# Edge Cases & Additional Tests
# ============================================================================


@patch("voicepad_core.audio.file.sf.read")
def test_path_as_string_or_pathlib(mock_sf_read: Mock, tmp_path: Path) -> None:
    """Test both str and Path inputs work correctly."""
    file_path = tmp_path / "test.wav"
    file_path.touch()

    mock_audio = np.array([0.1, 0.2], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    # Test with Path object
    source1 = FileSource(file_path)
    result1 = source1.read()
    npt.assert_array_equal(result1, mock_audio)

    # Test with string
    source2 = FileSource(str(file_path))
    result2 = source2.read()
    npt.assert_array_equal(result2, mock_audio)


@patch("voicepad_core.audio.file.sf.read")
@patch("voicepad_core.audio.file.subprocess.run")
def test_temp_file_cleanup_on_success(mock_subprocess: Mock, mock_sf_read: Mock, tmp_path: Path) -> None:
    """Test temp file cleanup on successful conversion."""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.touch()

    mock_subprocess.return_value = Mock(returncode=0)
    mock_audio = np.array([0.1, 0.2], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 44100)

    # Track temp file creation
    with patch("voicepad_core.audio.file.tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp_file = Mock()
        mock_temp_file.name = str(tmp_path / "temp.wav")
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=False)
        mock_temp.return_value = mock_temp_file

        with (
            patch("voicepad_core.audio.file.os.path.exists", return_value=True),
            patch("voicepad_core.audio.file.os.remove") as mock_remove,
        ):
            source = FileSource(mp3_file)
            source.read()

            # Verify cleanup was called
            mock_remove.assert_called_once()


@patch("voicepad_core.audio.file.subprocess.run")
def test_ffmpeg_not_installed_raises_runtime_error(mock_subprocess: Mock, tmp_path: Path) -> None:
    """Test ffmpeg not installed raises RuntimeError."""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.touch()

    # Mock FileNotFoundError (ffmpeg not found)
    mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")

    source = FileSource(mp3_file)

    with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
        source.read()


@patch("voicepad_core.audio.file.sf.read")
def test_dtype_conversion_to_float32(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test audio is converted to float32 if needed."""
    # Mock soundfile returning int16
    mock_audio = np.array([100, 200, 300], dtype=np.int16)
    mock_sf_read.return_value = (mock_audio, 16000)

    source = FileSource(sample_wav_file)
    result = source.read()

    # Should be converted to float32
    assert result.dtype == np.float32
    npt.assert_array_equal(result, mock_audio.astype(np.float32))


@patch("voicepad_core.audio.file.sf.read")
def test_multichannel_detection(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test channel detection for various channel counts."""
    test_cases = [
        (1, np.array([0.1, 0.2], dtype=np.float32)),  # Mono 1D
        (2, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),  # Stereo
        (4, np.random.randn(10, 4).astype(np.float32)),  # 4-channel
        (6, np.random.randn(10, 6).astype(np.float32)),  # 5.1 surround
    ]

    for expected_channels, mock_audio in test_cases:
        mock_sf_read.return_value = (mock_audio, 48000)
        source = FileSource(sample_wav_file)
        source.read()
        assert source.get_channels() == expected_channels


@patch("voicepad_core.audio.file.sf.read")
def test_empty_audio_file(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test handling of empty audio file."""
    mock_audio = np.array([], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    source = FileSource(sample_wav_file)
    result = source.read()

    assert len(result) == 0
    assert result.dtype == np.float32
    assert source.get_channels() == 1


@patch("voicepad_core.audio.file.sf.read")
def test_very_short_audio(mock_sf_read: Mock, sample_wav_file: Path) -> None:
    """Test handling of very short audio (single sample)."""
    mock_audio = np.array([0.5], dtype=np.float32)
    mock_sf_read.return_value = (mock_audio, 16000)

    source = FileSource(sample_wav_file)
    result = source.read()

    assert len(result) == 1
    assert result[0] == pytest.approx(0.5)
