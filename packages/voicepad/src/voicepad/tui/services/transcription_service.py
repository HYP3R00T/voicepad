"""Transcription service for VoicePad TUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from voicepad_core import (
    TranscriptionError,
    ensure_model_downloaded,
    get_or_load_model,
    model_downloaded,
    transcribe_buffer,
)
from voicepad_core.transcription import AudioTooShortError

from voicepad.tui.workers import ModelWarmResult

if TYPE_CHECKING:
    from voicepad_core.config import Config
    from voicepad_core.transcription import TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for managing transcription operations."""

    def __init__(self, config: Config) -> None:
        """Initialize the transcription service.

        Args:
            config: Application configuration
        """
        self.config = config

    def warm_model(self) -> ModelWarmResult:
        """Download (if needed) and load the model into VRAM.

        This operation blocks until complete.

        Returns:
            ModelWarmResult containing device info or error
        """
        try:
            # Check if model needs to be downloaded
            if not model_downloaded(self.config.transcription_model, self.config):
                logger.info(f"Model '{self.config.transcription_model}' not cached — downloading")
                ensure_model_downloaded(self.config.transcription_model, self.config)

            # Load the model
            _, device, compute, fallback = get_or_load_model(self.config)
            logger.info(f"Model loaded: device={device}, compute={compute}, fallback={fallback}")
            return ModelWarmResult(device=device, compute_type=compute, fallback=fallback)

        except TranscriptionError as e:
            logger.error(f"Model download failed: {e}")
            return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))
        except Exception as e:
            logger.error(f"Model warm failed: {e}")
            return ModelWarmResult(device="cpu", compute_type="int8", fallback=True, error=str(e))

    def transcribe_audio(self, audio: np.ndarray) -> TranscriptionResult | None:
        """Transcribe an audio buffer.

        Args:
            audio: The audio data to transcribe

        Returns:
            TranscriptionResult if successful, None if error

        Raises:
            AudioTooShortError: If the audio is too short to transcribe
            TranscriptionError: If transcription fails
        """
        try:
            result = transcribe_buffer(audio, self.config)
            logger.info(f"Transcription complete: {len(result.text)} chars, {result.latency_ms:.0f}ms")
            return result
        except AudioTooShortError as e:
            logger.warning(f"Audio too short: {e}")
            raise
        except TranscriptionError as e:
            logger.error(f"Transcription failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected transcription error: {e}")
            raise

    def transcribe_file(self, wav_path: Path) -> TranscriptionResult | None:
        """Transcribe an audio file.

        Args:
            wav_path: Path to the WAV file to transcribe

        Returns:
            TranscriptionResult if successful, None if error

        Raises:
            FileNotFoundError: If the file doesn't exist
            AudioTooShortError: If the audio is too short to transcribe
            TranscriptionError: If transcription fails
        """
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        try:
            # Load the audio file
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            logger.info(f"Loaded audio file: {wav_path} ({len(audio)} samples, {sample_rate}Hz)")

            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                logger.warning(f"Resampling from {sample_rate}Hz to 16000Hz")
                # Simple resampling - for production, use a proper resampler
                import numpy as np

                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)),
                    np.arange(len(audio)),
                    audio,
                )

            # Transcribe the audio
            return self.transcribe_audio(audio)

        except (AudioTooShortError, TranscriptionError):
            raise
        except Exception as e:
            logger.error(f"Failed to transcribe file {wav_path}: {e}")
            raise
