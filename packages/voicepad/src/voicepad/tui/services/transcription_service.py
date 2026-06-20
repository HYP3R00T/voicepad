"""Transcription service for VoicePad TUI."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from voicepad_core import (
    TranscriptionError,
    begin_transcription_session,
    end_transcription_session,
    ensure_model_downloaded,
    load_model,
    log_transcription_end,
    log_transcription_start,
    model_downloaded,
    transcribe,
)
from voicepad_core.inference import AudioTooShortError

from voicepad.tui.workers import ModelWarmResult

if TYPE_CHECKING:
    from voicepad_core.config import Config
    from voicepad_core.inference import TranscriptionResult

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
        """Download (if needed) and load_model the model into VRAM.

        This operation blocks until complete.

        Returns:
            ModelWarmResult containing device info or error
        """
        try:
            # Check if model needs to be downloaded
            if not model_downloaded(self.config.transcription_model):
                logger.info(f"Model '{self.config.transcription_model}' not cached — downloading")
                ensure_model_downloaded(self.config.transcription_model)

            # Load the model
            model = load_model(
                self.config.transcription_model,
                self.config.transcription_device,
                self.config.transcription_compute_type,
            )
            device = self.config.transcription_device
            compute = self.config.transcription_compute_type
            fallback = False

            # Check if we fell back to CPU
            from voicepad_core.inference.model_manager import _model_cache

            for (m, d, c), cached_model in _model_cache.items():
                if m == self.config.transcription_model and cached_model is model:
                    device = d
                    compute = c
                    fallback = device == "cpu" and self.config.transcription_device != "cpu"
                    break

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
        # Set up per-transcription logging
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_logger, log_file = begin_transcription_session(
            logs_path=self.config.logs_path,
            log_level=self.config.log_level,
            session_id=session_id,
        )

        try:
            # Calculate audio duration
            duration_s = len(audio) / 16000  # Assuming 16kHz sample rate

            # Log transcription start
            log_transcription_start(
                session_logger,
                duration_s,
                self.config.transcription_model,
                self.config.transcription_device,
                self.config.transcription_compute_type,
            )

            # We call transcribe safely with expanded args
            result = transcribe(
                audio,
                model_name=self.config.transcription_model,
                device=self.config.transcription_device,
                compute_type=self.config.transcription_compute_type,
            )

            # Log transcription end
            log_transcription_end(
                session_logger,
                success=True,
                latency_ms=result.latency_ms,
                text_length=len(result.text),
            )

            logger.info(f"Transcription complete: {len(result.text)} chars, {result.latency_ms:.0f}ms")
            logger.info(f"Log file: {log_file}")
            return result

        except AudioTooShortError as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.warning(f"Audio too short: {e}")
            raise
        except TranscriptionError as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.error(f"Transcription failed: {e}")
            raise
        except Exception as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.error(f"Unexpected transcription error: {e}")
            raise
        finally:
            end_transcription_session()

    def transcribe_file(self, wav_path: Path) -> TranscriptionResult | None:
        """Transcribe an audio file using the new transcribe_file function.

        Args:
            wav_path: Path to the WAV file to transcribe

        Returns:
            TranscriptionResult if successful, None if error

        Raises:
            FileNotFoundError: If the file doesn't exist
            AudioTooShortError: If the audio is too short to transcribe
            TranscriptionError: If transcription fails
        """
        from voicepad_core import transcribe_file as core_transcribe_file

        # Set up per-transcription logging
        session_id = f"{wav_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_logger, log_file = begin_transcription_session(
            logs_path=self.config.logs_path,
            log_level=self.config.log_level,
            session_id=session_id,
        )

        try:
            session_logger.info(f"Transcribing file: {wav_path}")

            # Use the new transcribe_file function from voicepad-core
            result = core_transcribe_file(
                wav_path,
                model_name=self.config.transcription_model,
                device=self.config.transcription_device,
                compute_type=self.config.transcription_compute_type,
                language=self.config.language,
                local_agreement=self.config.local_agreement_file,
            )

            # Log transcription end
            log_transcription_end(
                session_logger,
                success=True,
                latency_ms=result.latency_ms,
                text_length=len(result.text),
            )

            logger.info(f"File transcription complete: {len(result.text)} chars, {result.latency_ms:.0f}ms")
            logger.info(f"Log file: {log_file}")
            return result

        except (AudioTooShortError, TranscriptionError) as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            raise
        except Exception as e:
            log_transcription_end(session_logger, success=False, error=str(e))
            logger.error(f"Failed to transcribe file {wav_path}: {e}")
            raise
        finally:
            end_transcription_session()
