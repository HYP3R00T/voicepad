"""Microphone capture for push-to-talk dictation.

The entire public API is three methods:

    recorder.start()          — open the mic, begin collecting samples
    audio = recorder.stop()   — close the mic, return float32 array at 16 kHz
    recorder.is_recording()   — True while mic is open

That array goes straight into transcribe_buffer(). Nothing else happens here.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soundfile as sf

if TYPE_CHECKING:
    from voicepad_core.config import Config

logger = logging.getLogger(__name__)

# Fixed for Whisper compatibility — never changes
SAMPLE_RATE: int = 16000
CHANNELS: int = 1


class AudioRecorderError(Exception):
    """Raised when the recorder cannot start or stop cleanly."""


class AudioRecorder:
    """Captures microphone audio as a numpy array.

    Example:
        recorder = AudioRecorder(config)
        recorder.start()
        # ... user speaks ...
        audio = recorder.stop()                    # float32, 16 kHz, mono
        result = transcribe_buffer(audio, config)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._ensure_recordings_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone and start collecting audio.

        Raises:
            AudioRecorderError: If already recording or the device cannot be opened.
        """
        if self._recording:
            raise AudioRecorderError("Already recording")

        self._frames = []

        try:
            self._stream = sd.InputStream(
                device=self.config.input_device_index,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise AudioRecorderError(f"Cannot open audio device: {e}") from e

        self._recording = True
        logger.info(f"Recording started (device={self.config.input_device_index})")

    def stop(self) -> np.ndarray:
        """Close the microphone and return all captured audio.

        Returns:
            float32 numpy array at 16 kHz mono.
            Empty array (length 0) if nothing was captured.

        Raises:
            AudioRecorderError: If not currently recording.
        """
        if not self._recording:
            raise AudioRecorderError("Not recording")

        self._recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
            finally:
                self._stream = None

        with self._lock:
            frames = list(self._frames)

        if not frames:
            logger.warning("No audio captured")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames).flatten()
        logger.info(f"Recording stopped — {len(audio) / SAMPLE_RATE:.2f}s captured")
        return audio

    def is_recording(self) -> bool:
        """Return True while the microphone is open."""
        return self._recording

    def save_wav(self, audio: np.ndarray, path: Path) -> None:
        """Save a float32 audio array to a 16-bit WAV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        logger.debug(f"Saved {len(audio) / SAMPLE_RATE:.2f}s → {path}")

    def make_wav_path(self, prefix: str | None = None) -> Path:
        """Return a timestamped WAV path under the recordings directory."""
        from datetime import datetime

        pfx = prefix or self.config.recording_prefix
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.recordings_path / f"{pfx}_{ts}.wav"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning(f"Stream status: {status}")
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())

    def _ensure_recordings_dir(self) -> None:
        path = self.config.recordings_path
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise AudioRecorderError(f"Cannot create recordings directory '{path}': {e}") from e
