from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from voicepad_core.audio.types import RawAudio
from voicepad_core.audio.wav_persistence import WavArtifact, write_wav_atomic


class TestWriteWavAtomic:
    """Tests for durable, atomic WAV persistence."""

    def test_success_returns_artifact_for_complete_wav(self, tmp_path: Path) -> None:
        """A successful write returns matching metadata and a readable complete WAV."""
        destination = tmp_path / "nested" / "recording.wav"
        samples = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
        audio = RawAudio(samples=samples, sample_rate=4, channels=1)

        artifact = write_wav_atomic(audio, destination)
        persisted_samples, persisted_rate = sf.read(destination, dtype="float32")

        assert (
            artifact,
            destination.exists(),
            persisted_rate,
            persisted_samples.shape,
        ) == (
            WavArtifact(
                path=destination,
                sample_rate=4,
                channels=1,
                frame_count=4,
                duration_s=1.0,
            ),
            True,
            4,
            samples.shape,
        )

    def test_write_failure_removes_temp_and_preserves_existing_destination(self, tmp_path: Path) -> None:
        """When encoding fails, no temporary file remains and an old recording is untouched."""
        destination = tmp_path / "recording.wav"
        destination.write_bytes(b"existing recording")
        audio = RawAudio(samples=np.zeros(4, dtype=np.float32), sample_rate=16_000, channels=1)

        with (
            patch("voicepad_core.audio.wav_persistence.sf.write", side_effect=RuntimeError("encode failed")),
            pytest.raises(RuntimeError, match="encode failed"),
        ):
            write_wav_atomic(audio, destination)

        assert (destination.read_bytes(), tuple(tmp_path.glob(".recording-*.wav"))) == (
            b"existing recording",
            (),
        )

    def test_promotion_failure_removes_temp_and_preserves_existing_destination(self, tmp_path: Path) -> None:
        """When no-overwrite publication fails, its temporary WAV is cleaned up."""
        destination = tmp_path / "recording.wav"
        destination.write_bytes(b"existing recording")
        audio = RawAudio(samples=np.zeros(4, dtype=np.float32), sample_rate=16_000, channels=1)

        with (
            patch("voicepad_core.audio.wav_persistence.os.link", side_effect=OSError("promotion failed")),
            pytest.raises(OSError, match="promotion failed"),
        ):
            write_wav_atomic(audio, destination)

        assert (destination.read_bytes(), tuple(tmp_path.glob(".recording-*.wav"))) == (
            b"existing recording",
            (),
        )
