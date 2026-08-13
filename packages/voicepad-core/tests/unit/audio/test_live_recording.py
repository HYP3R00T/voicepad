from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from voicepad_core.audio.errors import AudioStreamStateError
from voicepad_core.audio.live_recording import LiveWavRecording
from voicepad_core.audio.types import AudioWindow
from voicepad_core.audio.wav_persistence import WavArtifact


def test_live_recording_reads_sample_range_and_finalizes(tmp_path: Path) -> None:
    destination = tmp_path / "recording.wav"
    recording = LiveWavRecording(destination, sample_rate=4, channels=1)
    recording.start()
    recording.append(np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32))
    recording.append(np.array([-0.25, -0.5], dtype=np.float32))

    window = recording.read_from(2, max_samples=2)
    artifact = recording.finish()
    persisted, sample_rate = sf.read(destination, dtype="float32")

    assert (window.start_sample, window.end_sample) == (2, 4)
    np.testing.assert_allclose(window.samples, [0.5, 0.75])
    assert (artifact.frame_count, artifact.duration(), sample_rate, len(persisted)) == (6, 1.5, 4, 6)
    assert tuple(tmp_path.glob(".recording-live-*.wav")) == ()


def test_live_recording_refuses_overwrite_and_retains_recoverable_spool(tmp_path: Path) -> None:
    destination = tmp_path / "recording.wav"
    destination.write_bytes(b"existing recording")
    recording = LiveWavRecording(destination, sample_rate=16_000, channels=1)
    recording.start()
    recording.append(np.zeros(512, dtype=np.float32))

    with pytest.raises(FileExistsError):
        recording.finish()

    assert destination.read_bytes() == b"existing recording"
    assert len(tuple(tmp_path.glob(".recording-live-*.wav"))) == 1


def test_abort_retains_spool_if_writer_does_not_stop(tmp_path: Path) -> None:
    spool = tmp_path / ".recording-live.wav"
    spool.write_bytes(b"recoverable audio")
    recording = LiveWavRecording(tmp_path / "recording.wav", 16_000, 1)
    recording._thread = writer = MagicMock()
    recording._spool_path = spool
    writer.is_alive.return_value = True

    recording.abort()

    assert spool.exists()
    assert recording._thread is writer


def test_live_read_failure_does_not_stop_audio_persistence(tmp_path: Path) -> None:
    destination = tmp_path / "recording.wav"
    recording = LiveWavRecording(destination, sample_rate=4, channels=1)
    recording.start()
    recording.append(np.array([0.0, 0.25], dtype=np.float32))

    with (
        patch("voicepad_core.audio.live_recording._read_wav_window", side_effect=RuntimeError("read failed")),
        pytest.raises(AudioStreamStateError, match="Could not read live audio"),
    ):
        recording.read_from(0)

    recording.append(np.array([0.5, 0.75], dtype=np.float32))
    artifact = recording.finish()
    persisted, _ = sf.read(destination, dtype="float32")

    assert artifact.frame_count == 4
    np.testing.assert_allclose(persisted, [0.0, 0.25, 0.5, 0.75])


def test_live_recording_read_during_finalization_uses_finished_artifact(tmp_path: Path) -> None:
    """A read racing with finalization completes from the persisted artifact without timing out."""
    destination = tmp_path / "recording.wav"
    recording = LiveWavRecording(destination, sample_rate=4, channels=1)
    recording.start()
    recording.append(np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32))
    finalizing = Event()
    continue_finalizing = Event()
    from voicepad_core.audio.wav_persistence import _finalize_live_wav

    def delayed_finalize(
        spool_path: Path,
        final_path: Path,
        sample_rate: int,
        channels: int,
        frame_count: int,
    ) -> WavArtifact:
        finalizing.set()
        assert continue_finalizing.wait(timeout=1.0)
        return _finalize_live_wav(spool_path, final_path, sample_rate, channels, frame_count)

    finish_thread = Thread(target=recording.finish)
    with patch("voicepad_core.audio.live_recording._finalize_live_wav", side_effect=delayed_finalize):
        finish_thread.start()
        assert finalizing.wait(timeout=1.0)
        result: list[AudioWindow] = []
        read_thread = Thread(target=lambda: result.append(recording.read_from(1, max_samples=2)), daemon=True)
        read_thread.start()
        continue_finalizing.set()
        finish_thread.join(timeout=1.0)
        read_thread.join(timeout=1.0)

    assert not finish_thread.is_alive() and not read_thread.is_alive()
    assert len(result) == 1
    np.testing.assert_allclose(result[0].samples, [0.25, 0.5])
