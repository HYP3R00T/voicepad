from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from voicepad_core import MicrophoneStream, RawAudio, WavArtifact

if TYPE_CHECKING:
    from voicepad_core import TranscriptionResult
    from voicepad_core.config import Config

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
    """Prepare and activate the configured model, returning its actual runtime."""
    from voicepad_core import TranscriptionError, activate_model, model_is_ready, prepare_model

    try:
        if not model_is_ready(config.transcription_model):
            logger.info(f"Model '{config.transcription_model}' not cached — downloading")
            prepare_model(config.transcription_model)

        runtime = activate_model(
            config.transcription_model,
            config.transcription_device,
            config.transcription_compute_type,
        )
        return ModelWarmResult(
            device=runtime.device,
            compute_type=runtime.precision,
            fallback=runtime.fallback_to_cpu,
        )
    except TranscriptionError as e:
        logger.error("Model preparation or activation failed: %s", e)
        return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))
    except Exception as e:
        logger.error("Unexpected model warm failure: %s", e)
        return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))


# ---------------------------------------------------------------------------
# Recording worker
# ---------------------------------------------------------------------------


@dataclass
class RecordingSession:
    """Manages a live recording that can be stopped from another thread."""

    config: Config
    recording_path: Path | None = None
    _recorder: MicrophoneStream | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _error: str | None = field(default=None, init=False)

    def start(self) -> None:
        """Open the microphone. Call from a worker thread."""
        try:
            recording_path = self.recording_path or (
                self.config.recordings_path / f"{self.config.recording_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            )
            self.recording_path = recording_path
            self._recorder = MicrophoneStream(recording_path, device_index=self.config.input_device_index)
            self._recorder.start()
        except Exception as e:
            self._error = str(e)
            raise

    def stop(self) -> WavArtifact:
        """Close the microphone and return the persisted recording."""
        if self._recorder is None:
            raise RuntimeError("RecordingSession has not been started.")
        try:
            return self._recorder.stop()
        except Exception as e:
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
    """Transcribe canonical audio. Blocks until complete."""

    audio: RawAudio
    config: Config
    result: TranscriptionResult | None = field(default=None, init=False)
    error: str | None = field(default=None, init=False)

    def run(self) -> TranscriptionResult | None:
        from voicepad_core import AudioTooShortError, TranscriptionError, transcribe

        try:
            self.result = transcribe(
                self.audio,
                model_name=self.config.transcription_model,
                device=self.config.transcription_device,
                compute_type=self.config.transcription_compute_type,
                language=self.config.language,
                word_timestamps=False,
            )
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
