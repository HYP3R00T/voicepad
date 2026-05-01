"""Background workers for the VoicePad TUI.

All blocking operations (model load, audio recording, transcription) run in
worker threads so the Textual event loop stays responsive.

Each worker is a plain dataclass that holds its result or error after run().
The TUI calls worker.run() via app.run_worker() and reads the result in the
on_worker_state_changed handler.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from voicepad_core import AudioRecorder

if TYPE_CHECKING:
    from voicepad_core.config import Config
    from voicepad_core.transcription import TranscriptionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model warm-up worker
# ---------------------------------------------------------------------------


@dataclass
class ModelWarmResult:
    device: str
    compute_type: str
    fallback: bool
    error: str | None = None


def warm_model(config: Config) -> ModelWarmResult:
    """Download (if needed) then load the model into VRAM. Blocks until complete."""
    from voicepad_core import ensure_model_downloaded, get_or_load_model, model_downloaded
    from voicepad_core.transcription import TranscriptionError

    try:
        if not model_downloaded(config.transcription_model, config):
            logger.info(f"Model '{config.transcription_model}' not cached — downloading")
            ensure_model_downloaded(config.transcription_model, config)
        _, device, compute, fallback = get_or_load_model(config)
        return ModelWarmResult(device=device, compute_type=compute, fallback=fallback)
    except TranscriptionError as e:
        logger.error(f"Model download failed: {e}")
        return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))
    except Exception as e:
        logger.error(f"Model warm failed: {e}")
        return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))


# ---------------------------------------------------------------------------
# Recording worker
# ---------------------------------------------------------------------------


@dataclass
class RecordingSession:
    """Manages a live recording that can be stopped from another thread."""

    config: Config
    _recorder: AudioRecorder | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _audio: np.ndarray | None = field(default=None, init=False)
    _error: str | None = field(default=None, init=False)

    def start(self) -> None:
        """Open the microphone. Call from a worker thread."""
        from voicepad_core import AudioRecorderError

        try:
            self._recorder = AudioRecorder(self.config)
            self._recorder.start()
        except AudioRecorderError as e:
            self._error = str(e)
            raise

    def stop(self) -> np.ndarray:
        """Close the microphone and return captured audio."""
        from voicepad_core import AudioRecorderError

        if self._recorder is None:
            return np.array([], dtype=np.float32)
        try:
            audio = self._recorder.stop()
            self._audio = audio
            return audio
        except AudioRecorderError as e:
            self._error = str(e)
            raise

    @property
    def error(self) -> str | None:
        return self._error


# ---------------------------------------------------------------------------
# Transcription worker
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionJob:
    """Transcribes a numpy audio array. Blocks until complete."""

    audio: np.ndarray
    config: Config
    result: TranscriptionResult | None = field(default=None, init=False)
    error: str | None = field(default=None, init=False)

    def run(self) -> TranscriptionResult | None:
        from voicepad_core import TranscriptionError, transcribe_buffer
        from voicepad_core.transcription import AudioTooShortError

        try:
            self.result = transcribe_buffer(self.audio, self.config)
            return self.result
        except AudioTooShortError:
            self.error = "Recording too short — speak for at least 0.5 seconds"
            return None
        except TranscriptionError as e:
            self.error = str(e)
            return None
        except Exception as e:
            self.error = f"Unexpected error: {e}"
            return None
